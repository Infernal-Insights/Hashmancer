#!/usr/bin/env bash
set -e

SERVER=false
WORKER=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server)
      SERVER=true
      ;;
    --worker)
      WORKER=true
      ;;
    *)
      echo "Usage: $0 [--server] [--worker]"
      exit 1
      ;;
  esac
  shift
done

if ! $SERVER && ! $WORKER; then
  SERVER=true
  WORKER=true
fi

if $SERVER; then
  python3 setup.py --server
fi

if $WORKER; then
  python3 setup.py --worker
fi

echo
if $SERVER; then
  echo "Server installed. Start it with systemd or run:"
  echo "  uvicorn main:app --host 0.0.0.0 --port 8000"
fi

if $WORKER; then
  echo "Worker installed. Launch it with:"
  echo "  python -m hashmancer.worker.hashmancer_worker.worker_agent"
fi
