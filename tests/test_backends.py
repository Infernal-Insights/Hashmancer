from hashmancer.darkling.backends import get_supported_backends


def test_get_supported_backends():
    backends = get_supported_backends()
    assert isinstance(backends, dict)
    for key in backends:
        assert key in {"cuda", "hip", "opencl"}
