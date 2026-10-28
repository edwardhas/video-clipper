# syntax=docker/dockerfile:1
# Local Clipper image (the API and the worker run from the same image). Build from the repo root:
#   docker build -f deploy/Dockerfile -t local-clipper:$(git rev-parse --short HEAD) .
# API (default command):  python -m clipper.cli serve-api   (uvicorn, proxy headers from env)
# Worker:                 python -m clipper.worker

# Debian release of the runtime image: trixie ships FFmpeg 7.1 (bookworm ships 5.1, not verified with the renderer).
ARG DEBIAN_RELEASE=trixie

# ----------------------------------------------------------------------------- frontend build
FROM node:22-bookworm-slim AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ----------------------------------------------------------------------------- runtime
FROM python:3.12-slim-${DEBIAN_RELEASE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg/ffprobe from Debian (built with libass; the app renders captions itself and does not
# need it). ca-certificates: the OpenAI SDK's HTTP client uses the OS trust store. tini: PID 1
# that forwards signals so the worker can mark its job "interrupted" on shutdown.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

# JavaScript runtime that yt-dlp needs for YouTube (deno >= 2.3).
COPY --from=denoland/deno:bin-2.9.7 /deno /usr/local/bin/deno

# Runtime dependencies only, every package pinned and hash-checked.
COPY backend/requirements.lock /tmp/requirements.lock
RUN pip install --require-hashes -r /tmp/requirements.lock && rm /tmp/requirements.lock

RUN groupadd --system --gid 10001 clipper \
 && useradd --system --uid 10001 --gid clipper --home-dir /app --no-create-home --shell /usr/sbin/nologin clipper \
 && mkdir -p /data \
 && chown clipper:clipper /data

WORKDIR /app
COPY backend/pyproject.toml /app/backend/pyproject.toml
COPY backend/clipper /app/backend/clipper
COPY --from=frontend /src/frontend/dist /app/frontend/dist
RUN python -m compileall -q /app/backend/clipper

# Hosted mode by default: sign-in is mandatory and startup fails without secrets.
ENV PYTHONPATH=/app/backend \
    CLIPPER_MODE=hosted \
    CLIPPER_HOST=0.0.0.0 \
    CLIPPER_PORT=8765 \
    CLIPPER_DATA_DIR=/data \
    CLIPPER_FRONTEND_DIST=/app/frontend/dist

USER clipper
VOLUME ["/data"]
EXPOSE 8765

# GET /api/health with the public origin's Host header (python urllib, no curl needed).
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD ["python", "-m", "clipper.cli", "health", "--quiet"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "clipper.cli", "serve-api"]
