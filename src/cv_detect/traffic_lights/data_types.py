from enum import Enum
import av
from dataclasses import dataclass

import numpy as np


class Color(Enum):
    RED = 0
    AMBER = 1
    GREEN = 2


class HSV(Enum):
    HUE =  0
    SATURATION = 1
    LIGHTNESS = 2

class MinMax(Enum):
    MIN = 0
    MAX = 1

class Combine(Enum):
    AND = 0
    OR = 1
