import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.labelling.traffic_lights.data_types import (
    Color,
    HSVFilter
)


def detect_light_circles(img: np.ndarray,
                         color_filters: list[HSVFilter]):
    """First threshold the colour, then pick out small circles

    Args:
        img: Image to feed into detector
        color_filters: What values to use in the filter.
                      List with each filter's values combined with an OR.
    """
    assert len(color_filters) > 1, f"The color filter should have some filters in it"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = color_filters[0].get_mask(img)
    for filter_ in color_filters[1:]:
        mask += filter_.get_mask(img)


    #img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    plt.imshow(img)
    plt.show()

    #cv2.imshow("Color filtered", img)
    #cv2.waitKey(0)




