# CPU-only image: train / evaluate / predict without a GPU (the models are ~0.4 M params).
#   docker build -t chl-gapfill .
#   docker run --rm -v $PWD/data:/app/data -v $PWD/models:/app/models chl-gapfill train -c configs/convlstm.yaml --no-mlflow
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tini libhdf5-dev libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 app
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MLFLOW_DISABLE_AGENT_HINT=1
COPY pyproject.toml README.md LICENSE ./
# dependency layer (cached until pyproject.toml changes): install against a stub package, then drop
# the stub's in-tree build/ so its empty __init__.py cannot shadow the real sources below
RUN mkdir -p src/chl_gapfill && touch src/chl_gapfill/__init__.py \
    && pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[track]" \
    && rm -rf build src
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
RUN pip install --no-deps --force-reinstall . && rm -rf build && chown -R app:app /app
USER app
ENV CHL_ROOT=/app
ENTRYPOINT ["tini", "--", "chl-gapfill"]
CMD ["--help"]
