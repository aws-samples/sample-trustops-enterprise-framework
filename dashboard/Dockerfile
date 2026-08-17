# Pinned by digest so rebuilds are reproducible and a moved tag cannot swap
# the base image. Refresh deliberately to pick up base-image CVE fixes:
#   docker pull public.ecr.aws/docker/library/python:3.11-slim
#   docker inspect --format='{{index .RepoDigests 0}}' \
#       public.ecr.aws/docker/library/python:3.11-slim
FROM public.ecr.aws/docker/library/python:3.11.15-slim@sha256:94c50be2dc994b873b55bc123e95e6dbade08095b3dfd790f51c34de3f08cbb7

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as an unprivileged user. Without this the container runs as root, so a
# process compromise starts with full control of the container filesystem.
# Streamlit writes runtime state under $HOME, so the home directory has to be
# owned by that user.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 trustops \
    && chown -R trustops:trustops /app
USER trustops
ENV HOME=/home/trustops

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from dashboard.health import get_health_status; s=get_health_status(); exit(0 if s['status']=='healthy' else 1)"

ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
