import cv2
import numpy as np
import pytest

from tests.fixtures import FIXTURES_PATH

from src.data_types import Image

from src.cv_detect.traffic_lights import (
    detect_light_circles
)
from src.cv_detect.traffic_lights.data_types import (
    Color,
)


@pytest.fixture
def red_traffic_light():
    return Image(FIXTURES_PATH / "traffic_lights_red_front.jpg")


def test_traffic_red_light_detection(red_traffic_light):
    ret = detect_light_circles(red_traffic_light, Color.RED)

