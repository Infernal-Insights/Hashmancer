import ctypes.util
import subprocess


def get_supported_backends() -> dict[str, str | None]:
    """Return available GPU backends and driver versions when possible."""
    backends: dict[str, str | None] = {}
    if ctypes.util.find_library("cuda"):
        driver = None
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                driver = res.stdout.splitlines()[0].strip()
        except Exception:
            pass
        backends["cuda"] = driver
    if ctypes.util.find_library("amdhip64") or ctypes.util.find_library("hip_hcc"):
        backends["hip"] = None
    if ctypes.util.find_library("OpenCL"):
        backends["opencl"] = None
    return backends
