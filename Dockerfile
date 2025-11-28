# CVAT Queue Manager - Docker Image (Production)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY gunicorn.conf.py .
COPY app/ ./app/

# Create directories for data persistence with proper permissions
RUN mkdir -p /app/data /app/backups && \
    chmod 777 /app/data /app/backups

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run with Gunicorn + Uvicorn workers (production)
CMD ["gunicorn", "main:app", "-c", "gunicorn.conf.py"]
