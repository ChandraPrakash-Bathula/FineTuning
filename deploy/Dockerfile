# Docker Space: FastAPI serves the built lesson AND the inference API on one
# origin, so there is no CORS setup and no VITE_API_BASE to configure.
#
# Hardware: the bf16 base is ~15.2 GB and the 4-bit base ~4 GB.
#   LOAD_BASES=both   needs a 24 GB card (L4 / A10G)
#   LOAD_BASES=qlora  fits a 16 GB card (T4)   <- cheapest that works
# Set LOAD_BASES in the Space's Variables, and HF_TOKEN in Secrets if the
# weights repo is ever made private again.

# ---- stage 1: build the frontend --------------------------------------------
FROM node:20-slim AS web
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- stage 2: cuda runtime ---------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR $HOME/app

COPY --chown=user backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

COPY --chown=user backend/ ./backend/
# /api/registry reads this; it is the same file the frontend bundles.
COPY --chown=user data/registry.json ./data/registry.json
COPY --chown=user --from=web /build/dist ./frontend/dist

ENV FRONTEND_DIST=/home/user/app/frontend/dist \
    LOAD_BASES=both

EXPOSE 7860
CMD ["python3", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
