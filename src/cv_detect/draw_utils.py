import cv2
import numpy as np

from src.data_types import (
    Circle,
    Rectangle,
    Image,
)


def draw_circles(img: Image,
                 circles: tuple[Circle, ...],
                 *args,
                 **kwargs) -> Image:
    """Draws some circles on an image array -but doesn't show the image"""
    img_arr = img.arr.copy()
    if len(args) < 1:
        kwargs['color'] = kwargs.get('color', (255, 255, 0))
    if len(args) < 2:
        kwargs['thickness'] = kwargs.get('thickness', 6)

    for icircle, circle in enumerate(circles):
        cv2.circle(img_arr, circle.center, int(circle.radius), *args, **kwargs)

    return Image(img_arr)


def draw_rectangles(img: Image,
                    rectangles: tuple[Rectangle, ...],
                    *args,
                    **kwargs) -> Image:
    """Draws some rectangles on an image array -but doesn't show the image"""
    img_arr = img.arr.copy()
    if len(args) < 1:
        kwargs['color'] = kwargs.get('color', (255, 255, 0))
    if len(args) < 2:
        kwargs['thickness'] = kwargs.get('thickness', 6)

    for irectangle, rectangle in enumerate(rectangles):
        cv2.rectangle(img_arr, rectangle.pt1, rectangle.pt2, *args, **kwargs)

    return Image(img_arr)
