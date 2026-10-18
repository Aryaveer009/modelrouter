# ModelRouter proxy. Built by Aryaveer Mankotia (https://github.com/Aryaveer009)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY models ./models
RUN pip install --no-cache-dir ".[embeddings]" \
    && python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Config comes from env vars (see .env.example), e.g. docker run --env-file .env
EXPOSE 8000
CMD ["modelrouter", "serve", "--host", "0.0.0.0", "--port", "8000"]
