from pathlib import Path

import av

import pytest

from src.data_types import (
    Image,
)

FIXTURES_PATH = Path(__file__).parent


@pytest.fixture(scope="module")
def red_amber_traffic_light_360_video():
    with av.open(FIXTURES_PATH / "traffic_lights.360") as container:
        yield container


@pytest.fixture
def red_traffic_light():
    return Image(FIXTURES_PATH / "traffic_lights_red_front.jpg")


@pytest.fixture
def green_traffic_light():
    return Image(FIXTURES_PATH / "traffic_lights_green.jpg")


@pytest.fixture
def red_yellow_light():
    return Image(FIXTURES_PATH / "red_and_yellow.jpg")


