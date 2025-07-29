FROM python:3.11-slim

WORKDIR /app

# Install system packages required for building dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the repository
COPY . /app

# Install Python dependencies for the server
RUN pip install --no-cache-dir -r hashmancer/server/requirements.txt

# Expose the default uvicorn port
EXPOSE 8000

# Run the FastAPI server
WORKDIR /app/hashmancer/server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
