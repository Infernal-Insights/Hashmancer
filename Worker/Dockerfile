FROM python:3.11-slim

WORKDIR /app

# install hashcat
RUN apt-get update && apt-get install -y --no-install-recommends \
    hashcat \
    && rm -rf /var/lib/apt/lists/*

# copy worker package and install requirements
COPY hashmancer /app/hashmancer
COPY Worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["python", "-m", "hashmancer.worker.hashmancer_worker.worker_agent"]
