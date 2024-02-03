import cv2
import numpy as np
import av
import pytest

from src.go_pro.utils import split_360

from tests.fixtures import FIXTURES_PATH


@pytest.fixture
def red_amber_traffic_light_360():
    with av.open(FIXTURES_PATH / "traffic_lights.360") as container:
        yield container


def test_split_360(red_amber_traffic_light_360):
    img_360 =next(split_360(red_amber_traffic_light_360))
    assert img_360.front.shape == (1344, 1344, 3)
    assert np.isclose(np.mean(img_360.front), 106.59712294205877)

    assert img_360.front_left.shape == (1344, 688, 3)
    assert np.isclose(np.mean(img_360.front_left), 48.69298266484404)

    assert img_360.front_right.shape == (1344, 688, 3)
    assert np.isclose(np.mean(img_360.front_right), 135.13690476190476)

    assert img_360.rear.shape == (1344, 1344, 3)
    assert np.isclose(np.mean(img_360.rear), 116.83671513310185)

    assert img_360.rear_left.shape == (1344, 688, 3)
    assert np.isclose(np.mean(img_360.rear_left), 59.18303283038021)

    assert img_360.rear_right.shape == (1344, 688, 3)
    assert np.isclose(np.mean(img_360.rear_right), 135.0243304292405)

    assert img_360.front_top.shape == (688, 1344, 3)
    assert np.isclose(np.mean(img_360.front_top), 209.55505555843945)

    assert img_360.rear_top.shape == (688, 1344, 3)
    assert np.isclose(np.mean(img_360.rear_top), 192.667067169043)

    assert img_360.front_bottom.shape == (688, 1344, 3)
    assert np.isclose(np.mean(img_360.front_bottom), 62.08222122727482)

    assert img_360.rear_bottom.shape == (688, 1344, 3)
    assert np.isclose(np.mean(img_360.rear_bottom), 75.67842543085548)
