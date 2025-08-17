"""Darkling hash cracking engine Python interface."""

# Import modules individually with error handling
statistics = None
charsets = None
backends = None
launcher = None
gpu_helpers = None

try:
    from . import charsets
except ImportError:
    pass

try:
    from . import backends
except ImportError:
    pass

try:
    from . import launcher
except ImportError:
    pass

try:
    from . import gpu_helpers
except ImportError:
    pass

# Import statistics last due to dependencies
try:
    from . import statistics
except ImportError:
    pass

__all__ = ['statistics', 'charsets', 'backends', 'launcher', 'gpu_helpers']