import cv2
import numpy as np
import pytest
from pathlib import Path

from src.cv_detect.traffic_lights import (
    detect_light_circles,
    detect_black_traffic_light_structures,
)
from src.cv_detect.draw_utils import (
    draw_circles,
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

    assert ret_circles == [Circle(center=(422, 815), radius=10.404426574707031),
                           Circle(center=(1129, 539), radius=5.315173149108887),
                           Circle(center=(518, 520), radius=6.7362589836120605),
                           Circle(center=(1169, 492), radius=11.011457443237305)]


def test_traffic_yellow_red_light_detection(red_yellow_light):
    """Note: We actually get some red lights as well as the color overlaps"""
    ret_circles = detect_light_circles(red_yellow_light, Color.AMBER)

    assert ret_circles == [Circle(center=(2175, 1231), radius=10.965956687927246),
                            Circle(center=(3803, 1111), radius=19.163867950439453),
                            Circle(center=(3796, 1051), radius=18.861867904663086),
                            Circle(center=(3982, 928), radius=32.86725616455078),
                            Circle(center=(4007, 889), radius=12.90358829498291),
                            Circle(center=(3970, 840), radius=23.070642471313477)]


def test_traffic_green_light_detection(green_traffic_light):
    ret_circles = detect_light_circles(green_traffic_light, Color.GREEN)

    assert ret_circles == [Circle(center=(2901, 1164), radius=27.36762237548828)]


def test_detect_black_traffic_light_structures(red_traffic_light):
    bounding_boxes = detect_black_traffic_light_structures(red_traffic_light)


def test_traffic_light_360_video(red_amber_traffic_light_360_video):
    dir_ = Path('/Users/mattellis/Projects/GoProCam/storage/test')
    for iframe, frame_360 in enumerate(split_360(red_amber_traffic_light_360_video)):
        red_circles = detect_light_circles(frame_360.front, Color.RED)
        print(f"Saving {iframe}")
        draw_circles(frame_360.front,
                     red_circles
        ).save(dir_ / f"{str(iframe).zfill(4)}.jpg")
