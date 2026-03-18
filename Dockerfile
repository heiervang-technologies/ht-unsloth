# HT Unsloth Studio — GPU-enabled container with Studio pre-installed
# Build:  docker build -t ht-unsloth-studio .
# Run:    docker run --gpus all -p 8000:8000 ht-unsloth-studio

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

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

# Create venv
RUN python3.12 -m venv /opt/unsloth-venv
ENV PATH="/opt/unsloth-venv/bin:$PATH"
RUN pip install --upgrade pip

# Copy repo
WORKDIR /opt/unsloth
COPY . .

# Install unsloth + deps (torch will be pulled with CUDA support)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install -e ".[huggingface]"
RUN pip install bitsandbytes

# Install Studio Python deps
RUN cd studio && python install_python_stack.py

# Build frontend
RUN cd studio/frontend && npm install && npm run build
RUN cd studio/backend/core/data_recipe/oxc-validator && npm install

# Build llama.cpp with CUDA
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
