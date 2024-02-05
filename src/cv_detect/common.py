import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.data_types import Image
from src.cv_detect.traffic_lights.data_types import (
    Color,
    HSV,
    MinMax,
    Combine,
)


COLOR_THRESHOLDS = {
    Color.RED: {
        Combine.AND: {
            Combine.OR: {
                MinMax.MAX: {
                    HSV.HUE: 30
                },
                MinMax.MIN: {
                    HSV.HUE: 150,
                },
            },
            MinMax.MIN: {
                Combine.AND: {
                    HSV.SATURATION: 110,
                    HSV.LIGHTNESS: 220,
                },
            }
        }
    },
}


def apply_threshold(img_arr: np.ndarray,
                    thresholds: dict) -> np.ndarray:
    """Will apply a dictionary of thresholds to the image array.

    The keys of the thresholds dict can be:
        Combine.And, Combine.OR, MinMax.MIN, MinMax.MAX, HSV.HUE, HSV.SATURATION, HSV.LIGHTNESS.

    Args:
        img_arr: numpy array with all thresholds
        thresolds: dictionary with thresholds.

    Examples:
        apply_thresholds(img_arr,
                         {
                          Combine.AND: {
                             MinMax.MIN: {
                                 HSV.HUE: 30
                             },
                             MinMax.MAX: {
                                 HSV.HUE: 150
                             }
                          }
                         }
        )
    """
    for key, value in thresholds.items():
        if isinstance(key, (Combine)):
            if key == Combine.AND:
                mask = np.ones(img_arr.shape[:2], dtype=bool)
                for new_threshold in thresholds[key]:
                    mask = mask & apply_threshold(img_arr, new_threshold)
            elif key == Combine.OR:
                mask = np.zeros(img_arr.shape[:2], dtype=bool)
                for new_threshold in thresholds[key]:
                    mask = mask | apply_threshold(img_arr, new_threshold)
            return mask

        if isinstance(key, (MinMax)):
            assert isinstance(value, dict), f"Badly formatted thresholds dict. MinMax must have a dict entry: {value}"
            assert len(value) == 1, f"MinMax can only have 1 entry -if you want to combine multiple selects use 'and' or 'or': {value}"
            if key == MinMax.MIN:
                ind, val = next(iter(thresholds[MinMax.MIN].items()))
                return img_arr[:,:,ind.value] <= val
            elif key == MinMax.MAX:
                ind, val = next(iter(thresholds[MinMax.MAX].items()))
                return img_arr[:,:,ind.value] >= val
            else:
                raise ValueError(f"HSV must come after MinMax: {thresholds}")

