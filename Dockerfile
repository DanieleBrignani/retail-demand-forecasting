# One image for API, MLflow server, Prefect server/worker, exporter and one-shot pipeline commands.
# Dependencies come from requirements.lock (reproducible); the project is installed without re-resolving.
FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MLFLOW_DISABLE_AGENT_HINT=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock ./
RUN pip install -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
COPY dbt ./dbt
COPY configs ./configs
RUN useradd -m -u 10001 app && mkdir -p /data /artifacts && chown -R app /data /artifacts /app
USER app
ENV RETAILFC_HOME=/app RETAILFC_DATA_DIR=/data RETAILFC_ARTIFACTS_DIR=/artifacts
EXPOSE 8000
CMD ["retailfc", "api", "serve", "--host", "0.0.0.0", "--port", "8000"]
