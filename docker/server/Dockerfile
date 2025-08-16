# Hashmancer Server Dockerfile
FROM python:3.11-slim

# Set build argument
ARG BUILD_ENV=production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    BUILD_ENV=${BUILD_ENV}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY hashmancer/server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn[standard] \
    redis \
    psutil \
    prometheus-client

# Copy application code
COPY hashmancer/ /app/hashmancer/
COPY config/ /app/config/
COPY redis_tool.py /app/
COPY REDIS_IMPROVEMENTS.md /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/tmp

# Copy startup script
COPY docker/server/start-server.sh /app/start-server.sh
RUN chmod +x /app/start-server.sh

# Health check script
COPY docker/server/health-check.sh /app/health-check.sh
RUN chmod +x /app/health-check.sh

# Create non-root user
RUN useradd --create-home --shell /bin/bash hashmancer && \
    chown -R hashmancer:hashmancer /app
USER hashmancer

# Expose ports
EXPOSE 8080 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/health-check.sh

# Start command
CMD ["/app/start-server.sh"]