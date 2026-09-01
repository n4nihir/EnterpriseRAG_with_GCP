FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (curl for healthchecks, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv directly from the official image for fast & reliable package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy requirements and install via uv system-wide
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application source code
COPY . .

# Expose ports for FastAPI (8000), Streamlit RAG Chat (8501), and Streamlit Evals (8502)
EXPOSE 8000 8501 8502

# Default command (overridden per service in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]