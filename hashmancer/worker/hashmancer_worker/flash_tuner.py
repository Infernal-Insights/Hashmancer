import argparse
import json
from pathlib import Path

from .bios_flasher import apply_flash_settings, md5_speed

PRESETS_FILE = (
    Path(__file__).resolve().parents[3]
    / "hashmancer"
    / "server"
    / "flash_presets.json"
)


def load_presets() -> dict:
    if PRESETS_FILE.exists():
        try:
            with open(PRESETS_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def tune_power_limit(index: int, vendor: str, preset: dict) -> int:
    """Iteratively lower power_limit until performance drops."""
    baseline = md5_speed()
    power = int(preset.get("power_limit", 0))
    best = power
    step = 5
    while power >= 0:
        settings = {
            "vendor": vendor,
            "power_limit": power,
            "core_clock": preset.get("core_clock"),
            "mem_clock": preset.get("mem_clock"),
        }
        success = apply_flash_settings({"index": index}, settings)
        post = md5_speed() if success else 0
        if not success or post < baseline * 0.8:
            break
        best = power
        power -= step
    if best != int(preset.get("power_limit", 0)):
        apply_flash_settings(
            {"index": index},
            {
                "vendor": vendor,
                "power_limit": best,
                "core_clock": preset.get("core_clock"),
                "mem_clock": preset.get("mem_clock"),
            },
        )
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune GPU power limit")
    parser.add_argument("--index", type=int, default=0, help="GPU index")
    parser.add_argument("--vendor", choices=["nvidia", "amd"], default="nvidia")
    parser.add_argument("--model", type=str, help="GPU model name")
    args = parser.parse_args()

    presets = load_presets().get(args.vendor, {})
    model = (args.model or "").lower()
    preset = presets.get(model, presets.get("default", {}))
    best = tune_power_limit(args.index, args.vendor, preset)
    print(f"Stable power limit: {best}")


if __name__ == "__main__":
    main()

