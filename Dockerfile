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
RUN curl -fsSL https://build.openmodelica.org/omc/bootstrapkey.asc \
      | gpg --dearmor -o /usr/share/keyrings/openmodelica-keyring.gpg && \
    echo "deb [arch=arm64 signed-by=/usr/share/keyrings/openmodelica-keyring.gpg] \
      https://build.openmodelica.org/apt $(lsb_release -cs) release" \
      > /etc/apt/sources.list.d/openmodelica.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        omc \
        openmodelica \
    && rm -rf /var/lib/apt/lists/*

# ── CoolProp (shared library, ARM64) ─────────────────────────────────────────
# Build from source with shared library + PIC flags for FMU portability
RUN mkdir -p /build/CoolProp && \
    git clone --depth 1 --branch v${COOLPROP_VERSION} \
        https://github.com/CoolProp/CoolProp.git /build/CoolProp/src && \
    mkdir -p /build/CoolProp/build && \
    cmake -S /build/CoolProp/src \
          -B /build/CoolProp/build \
          -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DCOOLPROP_SHARED_LIBRARY=ON \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          -DCMAKE_INSTALL_PREFIX=/opt/libs/CoolProp \
          -DCOOLPROP_INSTALL_PREFIX=/opt/libs/CoolProp \
    && ninja -C /build/CoolProp/build -j$(nproc) \
    && ninja -C /build/CoolProp/build install \
    && cp /build/CoolProp/build/libCoolProp.so /opt/libs/CoolProp/ \
    # Python bindings
    && pip3 install --no-cache-dir CoolProp==${COOLPROP_VERSION} \
    && rm -rf /build/CoolProp/src

# ── ThermoPower & SCOPE Modelica libraries ────────────────────────────────────
# Clone into /opt/libs/ for loadFile() in OMPython scripts
RUN mkdir -p /opt/libs && \
    git clone --depth 1 \
        https://github.com/casella/ThermoPower.git \
        /opt/libs/ThermoPower && \
    git clone --depth 1 \
        https://github.com/sxwd4ever/UQSTEPS_modelica.git \
        /opt/libs/SCOPE

# ── ExternalMedia (ARM64, rpath-patched) ─────────────────────────────────────
# Must link against CoolProp and embed rpath=$ORIGIN for FMU portability
RUN mkdir -p /build/ExternalMedia && \
    git clone --depth 1 --branch v${EXTERNAL_MEDIA_VERSION} \
        https://github.com/modelica-3rdparty/ExternalMedia.git \
        /build/ExternalMedia/src && \
    mkdir -p /build/ExternalMedia/build && \
    cmake -S /build/ExternalMedia/src/ExternalMedia \
          -B /build/ExternalMedia/build \
          -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          -DCOOLPROP_INCLUDE_DIR=/build/CoolProp/src/include \
          -DCOOLPROP_LIBRARY=/opt/libs/CoolProp/libCoolProp.so \
          -DCMAKE_INSTALL_PREFIX=/opt/libs/ExternalMedia \
          -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,'\$ORIGIN'" \
          -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,'\$ORIGIN'" \
    && ninja -C /build/ExternalMedia/build -j$(nproc) \
    && ninja -C /build/ExternalMedia/build install \
    && cp /opt/libs/CoolProp/libCoolProp.so /opt/libs/ExternalMedia/lib/ \
    && cp -r /build/ExternalMedia/src/ExternalMedia/package.mo \
             /opt/libs/ExternalMedia/ \
    && rm -rf /build/ExternalMedia/src /build/CoolProp

# Collect all shared libraries and update ldconfig
RUN echo "/opt/libs/CoolProp" >> /etc/ld.so.conf.d/sco2rl.conf && \
    echo "/opt/libs/ExternalMedia/lib" >> /etc/ld.so.conf.d/sco2rl.conf && \
    ldconfig

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/pytorch:25.11-py3

ARG PYTHON_DEPS

# ── Copy compiled artifacts ───────────────────────────────────────────────────
COPY --from=builder /usr/bin/omc /usr/bin/omc
COPY --from=builder /usr/lib/omc/ /usr/lib/omc/
COPY --from=builder /usr/share/openmodelica/ /usr/share/openmodelica/
COPY --from=builder /opt/libs/ /opt/libs/

# ── Runtime system deps ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzmq3-dev \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# ── Dynamic linker config for .so files ──────────────────────────────────────
RUN echo "/opt/libs/CoolProp" >> /etc/ld.so.conf.d/sco2rl.conf && \
    echo "/opt/libs/ExternalMedia/lib" >> /etc/ld.so.conf.d/sco2rl.conf && \
    ldconfig

# ── Python packages ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir ${PYTHON_DEPS}

# ── Environment ───────────────────────────────────────────────────────────────
ENV MODELICAPATH=/opt/libs
ENV LD_LIBRARY_PATH=/opt/libs/CoolProp:/opt/libs/ExternalMedia/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/workspace/src
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "from OMPython import OMCSessionZMQ; import CoolProp.CoolProp as CP; CP.PropsSI('T','P',1e5,'Q',0,'CO2')" \
    || exit 1

# Default: drop into bash for interactive use
CMD ["/bin/bash"]
