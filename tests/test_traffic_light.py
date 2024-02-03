import cv2
import numpy as np
import pytest

from tests.fixtures import FIXTURES_PATH

from src.labelling.traffic_lights import (
    detect_light_circles
)
from src.labelling.traffic_lights.data_types import (
    Color,
    HSVFilter,
)


@pytest.fixture
def red_traffic_light():
    return cv2.imread(str(FIXTURES_PATH / "traffic_light_red.jpg"))


@pytest.fixture(scope='module')
def test_tiny_img():
    return np.array(
            [[[1, 1, 1],  #0
              [2, 2, 2],  #1
              [1, 2, 1],  #2
              [1, 1, 2],  #3
              [3, 1, 1],  #4
              [1, 4, 4],  #5
              [4, 1, 4],  #6
              [4, 4, 1]]] #7
    )


def test_hsv_filter_hue_min(test_tiny_img):
    filter1 = HSVFilter(hue_min=2)
    mask = filter1.get_mask(test_tiny_img.copy())
    assert (mask == [[0, 1, 0, 0, 1, 0, 1, 1]]).all()


def test_hsv_filter_hue_min_max(test_tiny_img):
    filter1 = HSVFilter(hue_min=2, hue_max=3)
    mask = filter1.get_mask(test_tiny_img.copy())
    assert (mask == [[0,1,0,0,1,0,0,0]]).all()


def test_hsv_filter_all_min_max(test_tiny_img):
    filter1 = HSVFilter(hue_min=2, hue_max=4,
                        saturation_min=2,
                        value_max=3)
    mask = filter1.get_mask(test_tiny_img.copy())
    assert (mask == [[0,1,0,0,0,0,0,1]]).all()


def test_traffic_red_light_detection(red_traffic_light):
    red_filter = [HSVFilter(hue_min=30, hue_max=150),
                  HSVFilter(value_max=150),
                  HSVFilter(saturation_max=20),]

    ret = detect_light_circles(red_traffic_light, red_filter)

