FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install runtime dependencies first for better layer caching.
COPY pyproject.toml /app/pyproject.toml
RUN pip install --upgrade pip && \
    pip install flask pyyaml requests

# Copy application source.
COPY . /app

EXPOSE 8000

# Basic container health probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["python", "main.py", "--serve", "--host", "0.0.0.0", "--port", "8000"]
