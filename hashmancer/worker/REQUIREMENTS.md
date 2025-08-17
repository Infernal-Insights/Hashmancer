# Requirements

The agent requires several system and Python packages.

System packages:
- `redis-server`
- `hashcat`
- NVIDIA drivers providing `nvidia-smi` or AMD's `rocm-smi`
- Intel GPUs work with standard `intel-gpu-tools`

Python packages are listed in `requirements.txt` and include pinned versions:
- `redis==6.2.0`
- `requests==2.32.4`
- `cryptography==45.0.5`

Running `python -m hashmancer_agent.setup_agent` will attempt to install
these packages automatically using `apt-get` and `pip`.
Ensure you have an internet connection and sufficient privileges.
