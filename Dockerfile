# Live engine image (NVIDIA GPU).
#   docker build -t freewidth .
#   docker run --gpus all -p 127.0.0.1:8000:8000 -e FREEWIDTH_API_KEY="$(openssl rand -hex 32)" \
#       -v $PWD/calibration:/app/calibration -v $HOME/.cache/huggingface:/home/app/.cache/huggingface \
#       freewidth
# The server listens on all interfaces inside the container, so it refuses to start without
# FREEWIDTH_API_KEY. Publish the port on 127.0.0.1 unless other machines must reach it.
FROM nvidia/cuda:13.5.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

WORKDIR /app
COPY requirements.txt requirements-gpu.txt ./
RUN pip3 install -r requirements-gpu.txt

COPY freewidth ./freewidth
COPY configs ./configs
COPY pyproject.toml README.md LICENSE ./
RUN pip3 install --no-deps . && mkdir -p /app/calibration && chown -R app:app /app/calibration

# never run the server as root
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
  CMD curl -fs http://127.0.0.1:8000/health || exit 1
ENTRYPOINT ["python3", "-m", "freewidth"]
CMD ["serve", "--config", "configs/engine_t4.yaml", "--host", "0.0.0.0", "--log-json"]
