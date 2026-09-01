FROM python:3.11-slim

# 1. Install uv directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 2. Copy and install dependencies first (leverages Docker layer cache)
COPY requirements.txt .

RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]