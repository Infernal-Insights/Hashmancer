import sys
from tests.test_helpers import install_stubs
install_stubs()
import hashmancer.server.main as main


def test_get_flash_settings_nvidia():
    info = main.get_flash_settings("GeForce RTX 3080")
    assert info["vendor"] == "nvidia"


def test_get_flash_settings_amd():
    info = main.get_flash_settings("Radeon RX 480")
    assert info["vendor"] == "amd"
