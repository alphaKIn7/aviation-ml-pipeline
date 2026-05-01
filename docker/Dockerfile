# ──────────────────────────────────────────────
# Dockerfile — Reproducible ML Environment
# ──────────────────────────────────────────────
#
# PURPOSE:
#   Package the entire ML pipeline into a Docker container so
#   that it runs IDENTICALLY on any machine — your laptop,
#   a teammate's laptop, or a cloud server.
#
# WHY DOCKER?
#   "It works on my machine" is NOT acceptable in production ML.
#   Docker ensures:
#   - Same Python version
#   - Same library versions
#   - Same OS-level dependencies
#   - Reproducible results
#
# STRUCTURE TO BUILD:
#   FROM python:3.11-slim
#   WORKDIR /app
#   COPY requirements.txt .
#   RUN pip install --no-cache-dir -r requirements.txt
#   COPY . .
#   CMD ["python", "scripts/train.py", "--config", "configs/train_config.yaml"]
#
# USAGE:
#   docker build -f docker/Dockerfile -t aviation-ml-pipeline .
#   docker run aviation-ml-pipeline
# ──────────────────────────────────────────────
