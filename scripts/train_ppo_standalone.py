#!/usr/bin/env python3
"""
Standalone PPO trainer for sCO2 surrogate environment.
Uses PyTorch directly with a simple, numerically stable PPO implementation.
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from sco2rl.surrogate.fno_model import SCO2SurrogateFNO
from sco2rl.surrogate.batched_env import TorchBatchedSurrogateEnv
from train_surrogate import build_env_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_ppo_standalone")

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden // 2, act_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs):
        return torch.tanh(self.mean_head(self.trunk(obs))), self.log_std.exp().clamp(1e-4, 2.0)

    def get_action(self, obs):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.sample()
        action = torch.tanh(raw)
        log_prob = (dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        return action, log_prob, mean

    def log_prob_actions(self, obs, actions):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        ac = actions.clamp(-0.9999, 0.9999)
        raw = torch.atanh(ac)
        lp = (dist.log_prob(raw) - torch.log(1 - ac.pow(2) + 1e-6)).sum(-1)
        ent = dist.entropy().sum(-1)
        return lp, ent

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, obs):
        return self.net(obs).squeeze(-1)

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T, N = rewards.shape
    advantages = np.zeros_like(rewards)
    last_adv = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        advantages[t] = last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
    return advantages, advantages + values[:-1]

def ppo_update(policy, value_fn, opt_p, opt_v, obs, actions, old_lp, advantages, returns,
               clip_eps=0.2, epochs=4, mini_batches=4, ent_coef=0.01, vf_coef=0.5, max_grad=0.5):
    B = obs.shape[0]
    mb = B // mini_batches
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    stats = {"pl": [], "vl": [], "ent": [], "kl": []}
    for _ in range(epochs):
        perm = torch.randperm(B, device=obs.device)
        for start in range(0, B, mb):
            idx = perm[start:start+mb]
            new_lp, ent = policy.log_prob_actions(obs[idx], actions[idx])
            ratio = torch.exp(new_lp - old_lp[idx])
            adv = advantages[idx]
            pl = -torch.min(ratio * adv, ratio.clamp(1-clip_eps, 1+clip_eps) * adv).mean()
            vl = nn.functional.mse_loss(value_fn(obs[idx]), returns[idx])
            loss = pl + vf_coef * vl - ent_coef * ent.mean()
            if not torch.isfinite(loss):
                continue
            opt_p.zero_grad(); opt_v.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad)
            nn.utils.clip_grad_norm_(value_fn.parameters(), max_grad)
            opt_p.step(); opt_v.step()
            with torch.no_grad():
                kl = ((ratio - 1) - torch.log(ratio + 1e-8)).mean()
            stats["pl"].append(pl.item()); stats["vl"].append(vl.item())
            stats["ent"].append(ent.mean().item()); stats["kl"].append(kl.item())
    return {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}

def train(args):
    device = torch.device(args.device)
    fno_cfg = yaml.safe_load(open("configs/surrogate/fno_surrogate.yaml"))
    env_cfg = yaml.safe_load(open("configs/environment/env.yaml"))
    env_config = build_env_config(env_cfg)
    nrm = np.load("artifacts/surrogate/best_fno_norm.npz", allow_pickle=True)
    n = 14
    env_config["normalization"] = {
        "obs_mean": nrm["x_mean"][0,:n,0].tolist(), "obs_std": nrm["x_std"][0,:n,0].tolist(),
        "act_mean": nrm["x_mean"][0,n:,0].tolist(), "act_std": nrm["x_std"][0,n:,0].tolist(),
        "next_obs_mean": nrm["y_mean"][0,:n,0].tolist(), "next_obs_std": nrm["y_std"][0,:n,0].tolist(),
    }
    fno_model = SCO2SurrogateFNO(config=fno_cfg["fno"])
    sd = torch.load(args.fno_checkpoint, weights_only=True, map_location=device)
    fno_model.fno.load_state_dict(sd, strict=False)
    fno_model.eval()
    env = TorchBatchedSurrogateEnv(model=fno_model, config=env_config, n_envs=args.n_envs, device=device)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]
    logger.info("n_envs=%d obs_dim=%d act_dim=%d", args.n_envs, obs_dim, act_dim)
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    value_fn = ValueNetwork(obs_dim).to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_v = torch.optim.Adam(value_fn.parameters(), lr=args.lr)
    R, N = args.rollout_steps, args.n_envs
    n_updates = args.total_transitions // (R * N)
    logger.info("PPO: %d updates x %d transitions", n_updates, R * N)
    obs_t = torch.tensor(env.reset(seed=42)[0], dtype=torch.float32, device=device)
    ep_rewards, ep_lengths = [], []
    cur_r = np.zeros(N); cur_l = np.zeros(N, dtype=int)
    total_steps = 0; t0 = time.time()
    for upd in range(n_updates):
        ro, ra, rlp, rr, rd, rv = [], [], [], [], [], []
        for _ in range(R):
            with torch.no_grad():
                act, lp, _ = policy.get_action(obs_t)
                val = value_fn(obs_t)
            obs_np, rew_np, term_np, trunc_np, _ = env.step(act.cpu().numpy())
            done_np = term_np | trunc_np
            ro.append(obs_t.clone()); ra.append(act.clone()); rlp.append(lp.clone())
            rr.append(torch.tensor(rew_np, dtype=torch.float32, device=device))
            rd.append(torch.tensor(done_np.astype(np.float32), device=device))
            rv.append(val.clone())
            cur_r += rew_np; cur_l += 1
            for i in np.where(done_np)[0]:
                ep_rewards.append(float(cur_r[i])); ep_lengths.append(int(cur_l[i]))
                cur_r[i] = 0.0; cur_l[i] = 0
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
            total_steps += N
        with torch.no_grad():
            last_val = value_fn(obs_t)
        rew_a = torch.stack(rr).cpu().numpy()
        don_a = torch.stack(rd).cpu().numpy()
        val_a = torch.stack(rv).cpu().numpy()
        all_v = np.concatenate([val_a, last_val.cpu().numpy()[None]], axis=0)
        adv, ret = compute_gae(rew_a, all_v, don_a)
        obs_b = torch.stack(ro).view(-1, obs_dim)
        act_b = torch.stack(ra).view(-1, act_dim)
        lp_b  = torch.stack(rlp).view(-1)
        adv_b = torch.tensor(adv.reshape(-1), dtype=torch.float32, device=device)
        ret_b = torch.tensor(ret.reshape(-1), dtype=torch.float32, device=device)
        s = ppo_update(policy, value_fn, opt_p, opt_v, obs_b, act_b, lp_b, adv_b, ret_b)
        if any(torch.isnan(p).any() for p in policy.parameters()):
            logger.error("NaN in policy at update %d! Stopping.", upd); break
        if (upd+1) % 20 == 0:
            fps = total_steps / (time.time() - t0)
            mr = float(np.mean(ep_rewards[-100:])) if ep_rewards else float("nan")
            ml = float(np.mean(ep_lengths[-100:])) if ep_lengths else float("nan")
            logger.info("Upd %d/%d | trans=%d | fps=%.0f | r=%.3f | len=%.1f | pl=%.4f vl=%.4f ent=%.4f kl=%.4f",
                        upd+1, n_updates, total_steps, fps, mr, ml,
                        s["pl"], s["vl"], s["ent"], s["kl"])
    mr = float(np.mean(ep_rewards[-100:])) if ep_rewards else float("nan")
    logger.info("Done: trans=%d | mean_reward=%.3f", total_steps, mr)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path = ckpt_dir / "standalone_ppo_weights.pt"
    torch.save({"policy": policy.state_dict(), "value": value_fn.state_dict()}, weights_path)
    manifest = {"model_weights": str(weights_path.resolve()), "vec_normalize_stats": None,
                "curriculum_phase": 0, "lagrange_multipliers": {}, "total_timesteps": int(total_steps),
                "mean_reward": float(mr), "framework": "standalone_ppo"}
    (ckpt_dir / "standalone_ppo_checkpoint.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Checkpoint saved to %s", ckpt_dir)
    return manifest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fno-checkpoint", default="artifacts/surrogate/best_fno.pt")
    p.add_argument("--checkpoint-dir", default="artifacts/checkpoints/surrogate_ppo")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-envs", type=int, default=1024)
    p.add_argument("--rollout-steps", type=int, default=16)
    p.add_argument("--total-transitions", type=int, default=2_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    train(p.parse_args())
