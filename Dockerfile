# syntax=docker/dockerfile:1.4
# sCO₂ RL — Multi-stage ARM64 build for NVIDIA DGX Spark (GB10 Grace Blackwell)
# Platform: linux/arm64  Image: sco2-rl:latest
# Stage 1 builds OMC + CoolProp + ExternalMedia from source.
# Stage 2 layers Python/ML stack on NVIDIA PyTorch base.

ARG COOLPROP_VERSION=6.6.0
ARG EXTERNAL_MEDIA_VERSION=4.0.0
ARG PYTHON_DEPS="OMPython>=3.5 fmpy>=0.3.21 stable-baselines3>=2.3 gymnasium>=0.29 \
    scipy>=1.11 h5py>=3.10 tensorboard>=2.16 jinja2>=3.1 pydantic>=2.0 \
    pyyaml>=6.0 scikit-learn>=1.4 pytest>=8.0 pytest-cov ruff hatchling"

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
    git clone --depth 1 \
        https://github.com/casella/ThermoPower.git \
        /opt/libs/ThermoPower && \
    git clone --depth 1 \
        https://github.com/sxwd4ever/UQSTEPS_modelica.git \
        /opt/libs/SCOPE

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
