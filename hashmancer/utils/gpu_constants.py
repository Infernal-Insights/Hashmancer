import os
import re

# Load shared GPU constants from gpu_shared_types.h
_pattern = re.compile(r"#define\s+(MAX_HASHES|MAX_MASK_LEN|MAX_RESULT_BUFFER)\s+(\d+)")

_constants = {}
_header_path = os.path.join(os.path.dirname(__file__), "..", "darkling", "gpu_shared_types.h")
with open(os.path.abspath(_header_path)) as fh:
    for line in fh:
        m = _pattern.search(line)
        if m:
            _constants[m.group(1)] = int(m.group(2))

MAX_HASHES = _constants["MAX_HASHES"]
MAX_MASK_LEN = _constants["MAX_MASK_LEN"]
MAX_RESULT_BUFFER = _constants["MAX_RESULT_BUFFER"]
