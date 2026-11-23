"""shellm: a tiny llama-compatible llm you own end to end, plus weight-space programming."""
from .config import PRESETS, ModelConfig, preset
from .model import Shellm

__all__ = ["ModelConfig", "PRESETS", "preset", "Shellm"]
__version__ = "0.1.0"
