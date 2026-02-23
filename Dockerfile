FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files and readme (hatchling requires readme for build)
COPY pyproject.toml uv.lock* README.md ./

# Install dependencies (no dev)
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# Copy application
COPY app ./app

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
