from enum import Enum
import av
from dataclasses import dataclass

import numpy as np


class Color(Enum):
    RED = 0
    AMBER = 1
    GREEN = 2


@dataclass
class Img360:
    front: None


@dataclass
class HSVFilter:
    hue_min: int | None = None
    hue_max: int | None = None
    saturation_min: int | None = None
    saturation_max: int | None = None
    value_min: int | None = None
    value_max: int | None = None

    def get_mask(self, img: np.ndarray) -> np.ndarray:
        """Get a mask of values that fall within the range of the filter.

        All values are combined with AND.
        """
        # Iterate over mins and maxes and apply them with ands to combine
        mask = np.ones((img.shape[0], img.shape[1]))
        for i, prefix in enumerate(('hue', 'saturation', 'value')):
            min_ = getattr(self, f'{prefix}_min')
            max_ = getattr(self, f'{prefix}_max')
            if min_ is not None:
                mask *= (img[:, :, i] >= min_)
            if max_ is not None:
                mask *= (img[:, :, i] <= max_)

        return mask


    def __post_init__(self):
        for max_val, prefix in (
                (179, 'hue'),
                (255, 'saturation'),
                (255, 'value')
                ):
            min_ = getattr(self, f'{prefix}_min')
            max_ = getattr(self, f'{prefix}_max')
            if min_ and min_ < 0:
                raise ValueError(f"{prefix}_min={min_}, it must be between 0, {max_} inclusive")
            if max_ and max_ > max_val:
                raise ValueError(f"{prefix}_max={max_}, it must be between 0, {max_} inclusive")

            if min_ == 0:
                setattr(self, f"{prefix}_min", None)
            if max_ == max_val:
                setattr(self, f"{prefix}_max", None)


RED_FILTERS = [HSVFilter(hue_min=30, hue_max=150),]

