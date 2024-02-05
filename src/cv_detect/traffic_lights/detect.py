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
    # Will actually create the mask
    def _create_mask(img_arr, thresholds):
        pass


    mask = np.ones((img_arr.shape[0], img_arr.shape[1]),
                   dtype=bool)
    import ipdb; ipdb.set_trace()
    for key, value in thresholds:
        if isinstance(key, HSV):
            return


def detect_light_circles(img: np.ndarray, color: Color):
    """First threshold the colour, then pick out circles of a color.

    This can be used to find lights in a trafic light.

    Args:
        img: Image to feed into detector
        color_filters: What values to use in the filter.
                      List with each filter's values combined with an OR.
    """
    img_arr = cv2.cvtColor(img.arr, cv2.COLOR_RGB2HSV)
    thresholds = COLOR_THRESHOLDS[color]

    # Apply the HSV thresholds
    img_arr = apply_threshold(img_arr, COLOR_THRESHOLDS[color])

    Image(img_arr).show()

    # Erode dilate to remove noise
    erosion_size = 3
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2 * erosion_size + 1, 2 * erosion_size + 1),

                                        (erosion_size, erosion_size))
    img_arr = cv2.erode(img_arr, element)
    img_arr = cv2.dilate(img_arr, element)

    # Find circle-ish contours
    img_arr = cv2.Canny(img_arr, 100, 200)
    contours, hierachy = cv2.findContours(img_arr, cv2.RETR_EXTERNAL, 2)
    circles = [cv2.minEnclosingCircle(cnt) for cnt in contours]
    for icircle, (center, radius) in enumerate(circles):
        center = tuple(map(int, center))
        cv2.circle(img.arr, center, int(radius*1.1), (0, 255, 0), 3)
        print(cv2.contourArea(contours[icircle]), np.pi*radius**2)


    Image(img.arr).show()
    Image(img_arr).show()





