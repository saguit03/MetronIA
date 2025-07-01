from dataclasses import dataclass

VERBOSE_LOGGING = False
TOLERANCE_MS = 0.15


@dataclass
class AudioAnalysisConfig:
    hop_length: int = 1024
    n_chroma: int = 12
    plot_dpi: int = 300
    tolerance_ms: float = TOLERANCE_MS
    round_decimals: int = 2
