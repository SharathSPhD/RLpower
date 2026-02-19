# sCO₂ RL — Multi-stage ARM64 build for NVIDIA DGX Spark (GB10 Grace Blackwell)
# Platform: linux/arm64  Image: sco2-rl:latest
# Stage 1 builds OMC + CoolProp + ExternalMedia from source.
# Stage 2 layers Python/ML stack on NVIDIA PyTorch base.

ARG COOLPROP_VERSION=6.6.0
ARG EXTERNAL_MEDIA_VERSION=4.0.0
ARG PYTHON_DEPS="OMPython>=3.5 fmpy>=0.3.21 stable-baselines3>=2.3 gymnasium>=0.29 \
    scipy>=1.11 h5py>=3.10 tensorboard>=2.16 jinja2>=3.1 pydantic>=2.0 \
    pyyaml>=6.0 scikit-learn>=1.4 pytest>=8.0 pytest-cov ruff hatchling skrl>=1.4 \
    nvidia-physicsnemo"

# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM arm64v8/ubuntu:22.04 AS builder

ARG COOLPROP_VERSION
ARG EXTERNAL_MEDIA_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    # OpenModelica dependencies
    ca-certificates \
    gnupg \
    lsb-release \
    # ZMQ (required by OMPython IPC)
    libzmq3-dev \
    # Python for build scripts
    python3 \
    python3-pip \
    python3-dev \
    # Compression tools
    unzip \
    gzip \
    && rm -rf /var/lib/apt/lists/*

# ── OpenModelica (ARM64) ───────────────────────────────────────────────────────
# Install from official OMC package repository for Ubuntu 22.04 ARM64
RUN curl -fsSL https://build.openmodelica.org/apt/openmodelica.asc \
      | gpg --dearmor -o /usr/share/keyrings/openmodelica-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/openmodelica-keyring.gpg] \
      https://build.openmodelica.org/apt $(lsb_release -cs) release" \
      > /etc/apt/sources.list.d/openmodelica.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        openmodelica \
    && rm -rf /var/lib/apt/lists/*

# ── CoolProp (shared library, ARM64) ─────────────────────────────────────────
# Build from source with shared library + PIC flags for FMU portability
# NOTE: FORCE_BITNESS_NATIVE disables the -m64/-m32 flag that CoolProp CMake
#       adds when CMAKE_SIZEOF_VOID_P==8 — valid on x86_64, invalid on ARM64.
RUN mkdir -p /build/CoolProp && \
    git clone --depth 1 --branch v${COOLPROP_VERSION} \
        --recurse-submodules --shallow-submodules \
        https://github.com/CoolProp/CoolProp.git /build/CoolProp/src && \
    mkdir -p /build/CoolProp/build && \
    cmake -S /build/CoolProp/src \
          -B /build/CoolProp/build \
          -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DCOOLPROP_SHARED_LIBRARY=ON \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          -DFORCE_BITNESS_NATIVE=ON \
          -DCMAKE_INSTALL_PREFIX=/opt/libs/CoolProp \
          -DCOOLPROP_INSTALL_PREFIX=/opt/libs/CoolProp \
    && ninja -C /build/CoolProp/build -j$(nproc) \
    && ninja -C /build/CoolProp/build install \
    && cp /build/CoolProp/build/libCoolProp.so /opt/libs/CoolProp/ \
    # Python bindings
    && pip3 install --no-cache-dir CoolProp==${COOLPROP_VERSION}
    # NOTE: /build/CoolProp/src is intentionally kept — reused for ExternalMedia build

# ── ThermoPower & SCOPE Modelica libraries ────────────────────────────────────
# Clone into /opt/libs/ for loadFile() in OMPython scripts
RUN mkdir -p /opt/libs && \
    for i in 1 2 3 4 5; do \
        git clone --depth 1 \
            https://github.com/casella/ThermoPower.git \
            /opt/libs/ThermoPower && break || \
        (echo "ThermoPower clone attempt $i failed, retrying..."; sleep 10); \
    done && \
    for i in 1 2 3 4 5; do \
        git clone --depth 1 \
            https://github.com/sxwd4ever/UQSTEPS_modelica.git \
            /opt/libs/SCOPE && break || \
        (echo "SCOPE clone attempt $i failed, retrying..."; sleep 10); \
    done

# ── SCOPE Modelica 4.x compatibility patches (ARM64, OMC 1.26+) ─────────────
# SCOPE was written for Modelica 3.2.1. Patch for Modelica 4.1.0 / OMC 1.26.2.
# 1. Bulk replace Modelica.SIunits namespace → Modelica.Units.SI
# 2. DataRecord.R → DataRecord.R_s (renamed in MSL 4.x)
# 3. Remove fixedX/reducedX from IdealGas (removed from MSL 4.x interface)
# 4. Rewrite Valve.mo (used undeclared medium_in/out.state)
# 5. Rewrite BaseExchanger.mo / Recuperator.mo to use PropsSI (not ExternalMedia C)
# 6. Build libMyProps.so (SCOPE's CoolProp wrapper) from stub C for ARM64
RUN SCOPE=/opt/libs/SCOPE/src/Modelica/Steps && \
    # --- 1. Namespace bulk replace ---
    find ${SCOPE} -name '*.mo' -exec \
        sed -i 's/Modelica\.SIunits\.Conversions/Modelica.Units.Conversions/g' {} + && \
    find ${SCOPE} -name '*.mo' -exec \
        sed -i 's/Modelica\.SIunits/Modelica.Units.SI/g' {} + && \
    # --- 2. DataRecord.R → DataRecord.R_s ---
    sed -i 's/\bR=188\.9244822140674\b/R_s=188.9244822140674/g' \
        ${SCOPE}/Media/SCO2.mo ${SCOPE}/Media/CO2.mo && \
    sed -i 's/\bdata\.R\b/data.R_s/g' ${SCOPE}/Media/CO2.mo && \
    # --- 3. Remove deprecated IdealGas parameters ---
    sed -i '/final fixedX = true,/d; /final reducedX = true,/d' \
        ${SCOPE}/Media/SCO2.mo ${SCOPE}/Media/CO2.mo && \
    echo "SCOPE Modelica 4.x namespace patches applied."

# Rewrite Valve.mo — original used undeclared medium_in/out.state
RUN cat > /opt/libs/SCOPE/src/Modelica/Steps/Components/Valve.mo << 'VALVE_EOF'
within Steps.Components;
model Valve "Isenthalpic throttle valve (pure fluid-port equations)"
  extends TwoPorts;
  parameter Modelica.Units.SI.AbsolutePressure p_outlet "Fixed outlet pressure";
equation
  outlet.m_flow + inlet.m_flow = 0;
  outlet.h_outflow = inStream(inlet.h_outflow);
  outlet.p = p_outlet;
  inlet.h_outflow = inStream(outlet.h_outflow);
end Valve;
VALVE_EOF

# Rewrite BaseExchanger.mo — remove PBMedia.BaseProperties that require ExternalMedia C
RUN cat > /opt/libs/SCOPE/src/Modelica/Steps/Components/BaseExchanger.mo << 'BX_EOF'
within Steps.Components;
model BaseExchanger "Base class for heat exchangers (port-only, no ExternalMedia dependency)"
  replaceable package PBMedia = Steps.Media.SCO2;
  replaceable Steps.Interfaces.PBFluidPort_a inlet_hot(redeclare package Medium = PBMedia)
    annotation(Placement(transformation(extent={{-110,40},{-90,60}})));
  replaceable Steps.Interfaces.PBFluidPort_b outlet_hot(redeclare package Medium = PBMedia)
    annotation(Placement(transformation(extent={{90,40},{110,60}})));
  replaceable Steps.Interfaces.PBFluidPort_a inlet_cold(redeclare package Medium = PBMedia)
    annotation(Placement(transformation(extent={{-110,-60},{-90,-40}})));
  replaceable Steps.Interfaces.PBFluidPort_b outlet_cold(redeclare package Medium = PBMedia)
    annotation(Placement(transformation(extent={{90,-60},{110,-40}})));
  parameter Boolean debug_mode = false;
end BaseExchanger;
BX_EOF

# Rewrite Recuperator.mo — use Steps.Utilities.CoolProp.PropsSI (ARM64-safe)
RUN cat > /opt/libs/SCOPE/src/Modelica/Steps/Components/Recuperator.mo << 'RECUP_EOF'
within Steps.Components;
model Recuperator "Counter-flow recuperator using PropsSI (no ExternalMedia C dependency)"
  extends Steps.Components.BaseExchanger;
  import CP = Steps.Utilities.CoolProp;
  Real eta(min=0, max=1) "Heat exchange effectiveness";
  Modelica.Units.SI.SpecificEnthalpy h_hot_in;
  Modelica.Units.SI.SpecificEnthalpy h_cold_in;
  Modelica.Units.SI.Temperature T_hot_in;
  Modelica.Units.SI.Temperature T_cold_in;
  Modelica.Units.SI.SpecificEnthalpy h_cold_at_Thot;
  Modelica.Units.SI.SpecificEnthalpy h_hot_at_Tcold;
  Real Q_max_hot;
  Real Q_max_cold;
  Real Q_actual;
equation
  h_hot_in  = inStream(inlet_hot.h_outflow);
  h_cold_in = inStream(inlet_cold.h_outflow);
  T_hot_in  = CP.PropsSI("T", "P", inlet_hot.p,  "H", h_hot_in,  PBMedia.mediumName);
  T_cold_in = CP.PropsSI("T", "P", inlet_cold.p, "H", h_cold_in, PBMedia.mediumName);
  h_cold_at_Thot = CP.PropsSI("H", "P", inlet_cold.p, "T", T_hot_in,  PBMedia.mediumName);
  h_hot_at_Tcold = CP.PropsSI("H", "P", inlet_hot.p,  "T", T_cold_in, PBMedia.mediumName);
  Q_max_hot  = inlet_hot.m_flow  * (h_hot_in - h_cold_at_Thot);
  Q_max_cold = inlet_cold.m_flow * (h_hot_at_Tcold - h_cold_in);
  Q_actual = eta * min(Q_max_hot, Q_max_cold);
  outlet_cold.p        = inlet_cold.p;
  outlet_cold.m_flow  + inlet_cold.m_flow = 0;
  outlet_cold.h_outflow = h_cold_in + Q_actual / inlet_cold.m_flow;
  inlet_cold.h_outflow  = inStream(outlet_cold.h_outflow);
  outlet_hot.p         = inlet_hot.p;
  outlet_hot.m_flow   + inlet_hot.m_flow = 0;
  outlet_hot.h_outflow  = h_hot_in - Q_actual / inlet_hot.m_flow;
  inlet_hot.h_outflow   = inStream(outlet_hot.h_outflow);
end Recuperator;
RECUP_EOF

# Build MyProps.so for ARM64 — thin CoolProp C-API wrapper
# (replaces x86-only precompiled libMyProps from SCOPE)
COPY scripts/myprops_stub.c /tmp/myprops_stub.c
RUN mkdir -p /opt/libs/SCOPE/src/Modelica/Steps/Resources/Library/aarch64-linux && \
    gcc -O2 -fPIC -shared \
        /tmp/myprops_stub.c \
        -L/opt/libs/CoolProp -lCoolProp \
        -Wl,-rpath,'$ORIGIN' \
        -o /opt/libs/SCOPE/src/Modelica/Steps/Resources/Library/aarch64-linux/libMyProps.so \
        -lm && \
    echo "libMyProps.so (ARM64 stub) built successfully."

# ── ExternalMedia (ARM64, self-contained shared lib) ─────────────────────────
# ExternalMedia 4.0.0 embeds CoolProp as OBJECT library (no external .so dep).
# Reuse /build/CoolProp/src (kept from the standalone CoolProp build) as the
# pre-populated externals/CoolProp.git — avoids a second network clone.
# FORCE_BITNESS_NATIVE propagates to the CoolProp ADD_SUBDIRECTORY build.
# The cmake installs the .so into src/Modelica/ExternalMedia/Resources/Library/.
RUN mkdir -p /build/ExternalMedia && \
    git clone --depth 1 --branch v${EXTERNAL_MEDIA_VERSION} \
        https://github.com/modelica-3rdparty/ExternalMedia.git \
        /build/ExternalMedia/src && \
    mkdir -p /build/ExternalMedia/src/externals && \
    mv /build/CoolProp/src /build/ExternalMedia/src/externals/CoolProp.git && \
    mkdir -p /build/ExternalMedia/build && \
    cmake -S /build/ExternalMedia/src/Projects \
          -B /build/ExternalMedia/build \
          -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          -DFORCE_BITNESS_NATIVE=ON \
          -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,'\$ORIGIN'" \
          -DCMAKE_CXX_FLAGS="-I/build/ExternalMedia/src/externals/CoolProp.git/externals/fmtlib/include" \
    && ninja -C /build/ExternalMedia/build -j$(nproc) \
    && ninja -C /build/ExternalMedia/build install \
    # Copy the Modelica package (contains Resources/Library/linux64/.so) to /opt/libs/
    && mkdir -p /opt/libs && \
       cp -r /build/ExternalMedia/src/Modelica/ExternalMedia /opt/libs/ExternalMedia \
    && rm -rf /build/ExternalMedia /build/CoolProp/build

# Collect all shared libraries and update ldconfig
RUN echo "/opt/libs/CoolProp" >> /etc/ld.so.conf.d/sco2rl.conf && \
    echo "/opt/libs/ExternalMedia/Resources/Library/linux64" >> /etc/ld.so.conf.d/sco2rl.conf && \
    echo "/opt/libs/SCOPE/src/Modelica/Steps/Resources/Library/aarch64-linux" >> /etc/ld.so.conf.d/sco2rl.conf && \
    ldconfig

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/pytorch:25.11-py3

ARG PYTHON_DEPS
ARG COOLPROP_VERSION

# ── Copy compiled artifacts (CoolProp + ExternalMedia + Modelica libs) ────────
# OMC is re-installed from apt (avoids fragile path copies + handles all deps)
COPY --from=builder /opt/libs/ /opt/libs/

# ── Runtime system deps + OpenModelica (re-install from apt for correct deps) ─
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        lsb-release \
        libzmq3-dev \
        libgfortran5 \
    && curl -fsSL https://build.openmodelica.org/apt/openmodelica.asc \
         | gpg --dearmor -o /usr/share/keyrings/openmodelica-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/openmodelica-keyring.gpg] \
         https://build.openmodelica.org/apt $(lsb_release -cs) release" \
         > /etc/apt/sources.list.d/openmodelica.list \
    && apt-get update && apt-get install -y --no-install-recommends \
         openmodelica \
    && rm -rf /var/lib/apt/lists/*

# ── Dynamic linker config for .so files ──────────────────────────────────────
RUN echo "/opt/libs/CoolProp" >> /etc/ld.so.conf.d/sco2rl.conf && \
    echo "/opt/libs/ExternalMedia/Resources/Library/linux64" >> /etc/ld.so.conf.d/sco2rl.conf && \
    ldconfig

# ── Python packages ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir ${PYTHON_DEPS} CoolProp==${COOLPROP_VERSION}

# ── Environment ───────────────────────────────────────────────────────────────
ENV MODELICAPATH=/opt/libs
ENV LD_LIBRARY_PATH=/opt/libs/CoolProp:/opt/libs/ExternalMedia/Resources/Library/linux64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/workspace/src
ENV PYTHONDONTWRITEBYTECODE=1

# ── Non-root user (required: OMC ZMQ server refuses to start as root) ────────
RUN groupadd -g 1001 sco2rl && useradd -u 1001 -g sco2rl -m -s /bin/bash sco2rl \
    && mkdir -p /workspace && chown sco2rl:sco2rl /workspace \
    && chown -R sco2rl:sco2rl /opt/libs

# ── Pre-install Modelica Standard Library (baked into image as sco2rl user) ──
# OMC downloads MSL 4.1.0 to /home/sco2rl/.openmodelica/libraries/ on first
# installPackage() call. Running here as sco2rl ensures the path is correct.
USER sco2rl
RUN python3 -c "\
from OMPython import OMCSessionZMQ; \
omc = OMCSessionZMQ(); \
print('Installing Modelica MSL...', flush=True); \
r = omc.sendExpression('installPackage(Modelica)'); \
print('installPackage result:', r); \
r2 = omc.sendExpression('loadModel(Modelica)'); \
print('loadModel result:', r2); \
omc.sendExpression('quit()') \
"
USER root

WORKDIR /workspace

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "from OMPython import OMCSessionZMQ; import CoolProp.CoolProp as CP; CP.PropsSI('T','P',1e5,'Q',0,'CO2')" \
    || exit 1

# Default: drop into bash for interactive use
CMD ["/bin/bash"]
