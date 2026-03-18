# HT Unsloth Studio — GPU-enabled container with Studio pre-installed
#
# Build args:
#   CUDA_TAG     - base image CUDA version (default: 12.8.0)
#   TORCH_CUDA   - PyTorch CUDA wheel suffix (default: cu128)
#
# Examples:
#   docker build -t ht-unsloth-studio:cu128 .
#   docker build -t ht-unsloth-studio:cu130 --build-arg CUDA_TAG=13.0.0 --build-arg TORCH_CUDA=cu130 .

ARG CUDA_TAG=12.8.0
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu24.04

ARG TORCH_CUDA=cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface

# System deps + Node.js (for frontend build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git curl cmake ninja-build build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create outer venv (for unsloth CLI)
RUN python3.12 -m venv /opt/unsloth-venv
ENV PATH="/opt/unsloth-venv/bin:$PATH"
RUN pip install --upgrade pip

# Copy repo
WORKDIR /opt/unsloth
COPY . .

# Install unsloth CLI + deps (torch pinned to matching CUDA version)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_CUDA}
RUN pip install -e ".[huggingface]"
RUN pip install bitsandbytes

# Create Studio .venv and install deps (mirrors setup.sh lines 230-241)
RUN python3.12 -m venv /opt/unsloth/.venv \
    && /opt/unsloth/.venv/bin/python -m ensurepip --upgrade \
    && /opt/unsloth/.venv/bin/python -m pip install --upgrade pip \
    && PATH="/opt/unsloth/.venv/bin:$PATH" /opt/unsloth/.venv/bin/python studio/install_python_stack.py

# Pin torch in Studio .venv to match container CUDA version
# (install_python_stack pulls latest torch which may target a newer CUDA)
RUN /opt/unsloth/.venv/bin/python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_CUDA}

# Pre-install transformers 5.x (mirrors setup.sh lines 243-252)
RUN mkdir -p /opt/unsloth/.venv_t5 \
    && /opt/unsloth/.venv/bin/python -m pip install --target /opt/unsloth/.venv_t5 --no-deps \
       "transformers==5.3.0" "huggingface_hub==1.3.0"

# Build frontend
RUN cd studio/frontend && npm install && npm run build
RUN cd studio/backend/core/data_recipe/oxc-validator && npm install

# Build llama.cpp with CUDA (needs explicit -lcuda for Docker build without GPU)
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp \
    && cmake -G Ninja -S /opt/llama.cpp -B /opt/llama.cpp/build \
       -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
       -DGGML_CUDA=ON -DCMAKE_CUDA_FLAGS=--threads=0 \
       -DCMAKE_EXE_LINKER_FLAGS="-lcuda" -DCMAKE_SHARED_LINKER_FLAGS="-lcuda" \
    && cmake --build /opt/llama.cpp/build --config Release --target llama-server -j$(nproc) \
    && cmake --build /opt/llama.cpp/build --config Release --target llama-quantize -j$(nproc) || true \
    && ln -sf /opt/llama.cpp/build/bin/llama-quantize /opt/llama.cpp/llama-quantize 2>/dev/null || true

ENV UNSLOTH_LLAMA_CPP_DIR=/opt/llama.cpp

# Data volume for models/datasets
VOLUME /data

EXPOSE 8000

# Launch Studio
CMD ["unsloth", "studio", "-H", "0.0.0.0", "-p", "8000", "-f", "/opt/unsloth/studio/frontend/dist"]
