FROM python:3.11-slim

WORKDIR /app

# Install system packages and Node.js for frontend building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    git \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy the repository
COPY . /app

# Build React frontend
WORKDIR /app/hashmancer/frontend
RUN npm ci --only=production && npm run build

# Copy built frontend to server static directory
RUN mkdir -p /app/hashmancer/server/static && \
    cp -r dist/* /app/hashmancer/server/static/

# Install Python dependencies for the server
WORKDIR /app
RUN pip install --no-cache-dir -r hashmancer/server/requirements.txt

# Expose the default uvicorn port
EXPOSE 8000

# Run the FastAPI server
WORKDIR /app/hashmancer/server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
