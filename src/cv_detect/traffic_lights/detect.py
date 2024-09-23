import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.data_types import (
    Circle,
    Combine,
    Color,
    HSV,
    Image,
    MinMax,
)
from src.cv_detect.common import apply_threshold


COLOR_THRESHOLDS = {
    Color.RED: {
        Combine.OR: (
            {Combine.AND: (
                {MinMax.MIN: {HSV.HUE: 150}},
                {MinMax.MAX: {HSV.HUE: 50}},
            )},
            {MinMax.MIN: {HSV.SATURATION: 100}},
            {MinMax.MIN: {HSV.LIGHTNESS: 170}},
        ),
    },
    Color.GREEN: {
        Combine.OR: (
            {MinMax.MIN: {HSV.HUE: 60}},
            {MinMax.MAX: {HSV.HUE: 160}},
            {MinMax.MIN: {HSV.SATURATION: 100}},
            {MinMax.MIN: {HSV.LIGHTNESS: 170}},
        ),
    },
    Color.AMBER: {
        Combine.OR: (
            {MinMax.MIN: {HSV.HUE: 15}},
            {MinMax.MAX: {HSV.HUE: 65}},
            {MinMax.MIN: {HSV.SATURATION: 110}},
            {MinMax.MIN: {HSV.LIGHTNESS: 220}},
        ),
    },
    Color.BLACK: {
        MinMax.MAX: {HSV.LIGHTNESS: 70},
    }
}

def detect_traffic_lights(img: Image):
    """Will attempt to detect traffic lights by:

    1) Find black shapes
    2) Find red, amber or green circles within the black shape
    """


def detect_black_traffic_light_structures(img: Image):
    """Will filter for near-black colors and then return the bounding box of the shape"""
    # img_arr = cv2.bilateralFilter(img.arr, 15, 75, 95)
    img_arr = img.arr.copy()
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    # img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
    img_arr[img_arr > 25] = 255

    # Erode dilate to remove noise
    erosion_size=13
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, erosion_size))
    img_arr = cv2.dilate(img_arr, element)
    img_arr = cv2.erode(img_arr, element)

    erosion_size=3
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, 1))
    img_arr = cv2.dilate(img_arr, element)
    img_arr = cv2.erode(img_arr, element)

    # Get vertical lines
    # img_arr = cv2.Canny(img_arr, 70, 200)
    # lines = cv2.HoughLinesP(img_arr, 1, np.pi/180, 30, minLineLength=130, maxLineGap=40)
    # for line in lines:
    #     x1,y1,x2,y2 = line[0]
    #     cv2.line(img.arr,(x1,y1),(x2,y2),(0,255,0),2)

    import ipdb; ipdb.set_trace()


def mask_color(img: Image,
               color: Color,
               new_color: np.ndarray=0) -> np.ndarray:
    """Will set everything to that isn't the color in question to new_color

    Args:
        img: ...
        color: Name of colour
        new_color: Value of new color
    """
    img_arr = cv2.bilateralFilter(img.arr, 11, 75, 95)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)

    mask = apply_threshold(img_arr, COLOR_THRESHOLDS[color])
    img_arr[mask] = new_color
    return img_arr


def detect_light_circles(img: Image,
                         color: Color
    ) -> tuple[Circle, ...]:
    """First threshold the colour, then pick out circles of a color.

    This can be used to find lights in a trafic light.

    Args:
        img: Image to feed into detector
        color_filters: What values to use in the filter.
                      List with each filter's values combined with an OR.
    """
    # Set everything that isn't the color requested to black
    img_arr = mask_color(img, color)

    # Erode dilate to remove noise
    erosion_size = 3
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2*erosion_size + 1, 2*erosion_size + 1),
                                        (erosion_size, erosion_size))
    img_arr = cv2.erode(img_arr, element)
    img_arr = cv2.dilate(img_arr, element)

    # Find circle-ish contours
    img_arr = cv2.Canny(img_arr, 100, 200)
    contours, hierachy = cv2.findContours(img_arr, cv2.RETR_EXTERNAL, 2)

    circles = [cv2.minEnclosingCircle(cnt) for cnt in contours]
    ret_circles = []
    for icircle, (center, radius) in enumerate(circles):
        area = cv2.contourArea(contours[icircle])
        circle_amount = area / (np.pi*radius**2)
        if area > 10:
            center = tuple(map(int, center))

            # Some specific filtering for Amber -as it's very similar to red
            mask = cv2.circle(np.zeros(img.shape, np.uint8),
                              center=center,
                              radius=int(radius),
                              thickness=-1,
                              color=255)[:,:,0] > 1
            circle_cutout =  img.arr[mask]

            # Remove dark colors
            circle_cutout = circle_cutout[np.linalg.norm(circle_cutout, axis=1) > 100]
            avg_color = circle_cutout.mean(axis=0)
            std_color = circle_cutout.std(axis=0).mean()
            #if std_color < 50:
            ret_circles.append(Circle(radius=radius, center=center))

    return ret_circles
