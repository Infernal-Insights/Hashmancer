# Requirements

The agent requires several system and Python packages.

System packages:
- `redis-server`
- `hashcat`
- NVIDIA drivers providing `nvidia-smi`

Python packages are listed in `requirements.txt` and include:
- `redis`
- `requests`
- `cryptography`

Running `python -m hashmancer_agent.setup_agent` will attempt to install
these packages automatically using `apt-get` and `pip`.
Ensure you have an internet connection and sufficient privileges.
