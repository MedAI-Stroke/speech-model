import numpy as np
import parselmouth
from dataclasses import dataclass

@dataclass
class AudioObject:
    y: np.ndarray = None
    sr: int = 16000
    snd: parselmouth.Sound = None