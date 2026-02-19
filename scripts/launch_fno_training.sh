#!/usr/bin/env bash
# Auto-launcher: polls for collection completion, then starts FNO training.
# Usage: bash scripts/launch_fno_training.sh [--device cuda]
# Runs on the host; launches a Docker container for FNO training.

set -euo pipefail

DEVICE="${1:---device cuda}"
DEVICE_ARG="${DEVICE#--device=}"
DEVICE_ARG="${DEVICE:-cuda}"

DATASET="artifacts/trajectories/lhs_100k.h5"
EXPECTED_MIN_SIZE_GB=4.5     # Abort if dataset < 4.5 GB (incomplete)
LOG="artifacts/logs/fno_train_100k.log"
CONTAINER="sco2rl-fno-train-100k"

echo "[launch_fno] Waiting for collection to complete..."
echo "[launch_fno] Polling ${DATASET} every 5 minutes..."

while true; do
    if [ ! -f "${DATASET}" ]; then
        echo "[launch_fno] ${DATASET} not found yet."
        sleep 300; continue
    fi

    # Check if collection container has exited (done)
    COLL_STATUS=$(docker inspect --format='{{.State.Status}}' sco2rl-collect100k 2>/dev/null || echo "absent")
    SIZE_BYTES=$(stat -c%s "${DATASET}" 2>/dev/null || echo 0)
    SIZE_GB=$(python3 -c "print(${SIZE_BYTES}/1e9)")

    echo "[launch_fno] Collection status=${COLL_STATUS}  dataset=${SIZE_GB:.2f} GB"

    if [ "${COLL_STATUS}" != "running" ] && [ "${COLL_STATUS}" != "absent" ]; then
        echo "[launch_fno] Collection container exited. Dataset size: ${SIZE_GB} GB"
        if python3 -c "exit(0 if ${SIZE_BYTES} > ${EXPECTED_MIN_SIZE_GB} * 1e9 else 1)"; then
            echo "[launch_fno] Dataset sufficiently large. Starting FNO training."
            break
        else
            echo "[launch_fno] Dataset too small (${SIZE_GB} GB < ${EXPECTED_MIN_SIZE_GB} GB). Aborting."
            exit 1
        fi
    fi
    sleep 300
done

# Kill any existing surrogate containers
docker rm -f "${CONTAINER}" 2>/dev/null || true

echo "[launch_fno] Starting PhysicsNeMo FNO training (200 epochs, GPU)..."
docker run -d \
    -v "$(pwd):/workspace" \
    --gpus all \
    --shm-size=32g \
    --name "${CONTAINER}" \
    sco2-rl-automation:latest \
    bash -c "cd /workspace && PYTHONPATH=/workspace/src python scripts/train_surrogate.py \
        --dataset ${DATASET} \
        --device cuda \
        --skip-rl \
        --verbose 1 \
        2>&1 | tee ${LOG}"

echo "[launch_fno] FNO training container started: ${CONTAINER}"
echo "[launch_fno] Monitor with: docker logs -f ${CONTAINER}"
echo "[launch_fno] Log file: ${LOG}"
