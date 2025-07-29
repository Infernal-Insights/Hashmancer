from dataclasses import dataclass

@dataclass
class GPUInfo:
    uuid: str
    model: str = ""
    pci_bus: str = ""
    memory_mb: int = 0
    index: int = 0
    pci_link_width: int = 0
    vendor: str | None = None
