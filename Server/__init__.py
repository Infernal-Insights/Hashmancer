"""Compatibility shim for importing Server.* modules."""

from importlib import import_module
import sys

_base = import_module("hashmancer.server")

# Re-export public attributes from hashmancer.server
for name in getattr(_base, "__all__", []):
    globals()[name] = getattr(_base, name)

# Map expected submodules
sys.modules.setdefault(__name__ + ".signing_utils", import_module("hashmancer.server.signing_utils"))
sys.modules.setdefault(__name__ + ".hashescom_client", import_module("hashmancer.server.hashescom_client"))
