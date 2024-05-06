from pathlib import Path
import pytest
import numpy as np

from src.data_types import Rectangle
from src.ml_detect.cars import Detect
from src.data_types import Image
from src.cv_detect.draw_utils import (
    draw_rectangles,
)

from tests.fixtures import FIXTURES_PATH


@pytest.fixture()
def dark_night_cars_frame() -> Image:
    return Image(FIXTURES_PATH / 'dark_night_cars.jpg')


def test_extract_cars_from_frame(dark_night_cars_frame):
    detect = Detect(dark_night_cars_frame)
    rectangles = []
    for car in detect.extract('car'):
        if car.conf < 0.4: continue
        rectangles.append(car.xyxy[0])
    test_val = {','.join(map(str, i)) for i in rectangles}
    ref_val = {'3480.5566,1410.5154,4307.86,1854.228',
               '0.5983559,1631.3151,1245.2098,2654.6245',
               '1291.5586,1451.0995,2010.0381,1870.4843',
               '3078.3684,1427.1486,3296.743,1616.731',}

    assert test_val == ref_val

