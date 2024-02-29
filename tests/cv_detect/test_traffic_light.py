import cv2
import numpy as np
import pytest
from pathlib import Path

from src.cv_detect.traffic_lights import (
    detect_light_circles
)
from src.cv_detect.common import (
    draw_circles
)
from src.data_types import (
    Circle,
    Color,
)
from src.go_pro.utils import split_360

from tests.fixtures import (
    red_traffic_light,
    red_yellow_light,
    green_traffic_light,
    red_amber_traffic_light_360_video,
)


def test_traffic_red_light_detection(red_traffic_light):
    ret_circles = detect_light_circles(red_traffic_light, Color.RED)

    assert ret_circles == [Circle(radius=4.242740631103516, center=(1130, 538)),
                           Circle(radius=6.403224468231201, center=(518, 521)),
                           Circle(radius=10.404426574707031, center=(1169, 492))]


def test_traffic_yellow_red_light_detection(red_yellow_light):
    """Note: We actually get some red lights as well as the color overlaps"""
    ret_circles = detect_light_circles(red_yellow_light, Color.AMBER)
    assert ret_circles == [Circle(center=(2174, 1231), radius=11.335884094238281),
                           Circle(center=(3804, 1111), radius=18.681640625),
                           Circle(center=(3982, 928), radius=32.40811538696289)]


def test_traffic_green_light_detection(green_traffic_light):
    ret_circles = detect_light_circles(green_traffic_light, Color.GREEN)
    assert ret_circles == [Circle(center=(2901, 1165), radius=25.981271743774414)]


def test_traffic_light_360_video(red_amber_traffic_light_360_video):
    dir_ = Path('/Users/mattellis/Projects/GoProCam/storage/test')
    for iframe, frame_360 in enumerate(split_360(red_amber_traffic_light_360_video)):
        #if iframe != 4:
        #    continue
        red_circles = detect_light_circles(frame_360.front, Color.RED)
        print(f"Saving {iframe}")
        draw_circles(frame_360.front,
                     red_circles
        ).save(dir_ / f"{str(iframe).zfill(4)}.jpg")
