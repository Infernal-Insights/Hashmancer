import requests
import os

HASHES_API = os.environ.get("HASHES_COM_API_KEY")


def fetch_jobs():
    try:
        url = "https://hashes.com/en/api/jobs"
        if HASHES_API:
            url += f"?key={HASHES_API}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            return []
        return data["list"]
    except requests.HTTPError as e:
        print(f"[❌] Hashes.com fetch error: {e}")
        return []
    except Exception as e:
        print(f"[❌] Hashes.com fetch error: {e}")
        return []


def upload_founds(algo_id, found_file) -> bool:
    """Upload found hashes and return True if the API confirms success."""
    try:
        url = "https://hashes.com/en/api/founds"
        with open(found_file, "rb") as fh:
            files = {"userfile": fh}
            data = {"algo": str(algo_id), "key": HASHES_API or ""}
            resp = requests.post(url, files=files, data=data, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if not result.get("success"):
            print(f"[❌] Upload rejected: {result}")
            return False
        return True
    except requests.HTTPError as e:
        print(f"[❌] Upload error: {e}")
        return False
    except Exception as e:
        print(f"[❌] Upload error: {e}")
        return False
