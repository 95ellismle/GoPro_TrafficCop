import numpy as np
import pytest

from src.data_types import (
    HSV,
    MinMax,
    Combine,
)
from src.cv_detect.common import apply_threshold


@pytest.fixture
def img_arr():
    img_arr = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
               [[5, 5, 5], [6, 6, 6], [7, 7, 7]],
               [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
               [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
               ]
    return np.array(img_arr)


def test_simple_threshold(img_arr):
    # Just Min
    mask = apply_threshold(img_arr, {MinMax.MIN: {HSV.HUE: 3}})
    assert mask[[0, 2, 3]].all()
    assert not mask[[1]].any()


def test_bad_threshold(img_arr):
    # Badly defined
    fails = False
    try:
        apply_threshold(img_arr, {MinMax.MIN: {HSV.HUE: 3, HSV.SATURATION: 2}})
    except AssertionError as e:
        fails=True
    if not fails:
        raise ValueError("Expected to fail the apply_threshold test 2")


def test_and_threshold(img_arr):
    # Just and
    mask = apply_threshold(img_arr,
                           {Combine.AND: (
                                {MinMax.MIN: {HSV.HUE: 3}},
                                {MinMax.MIN: {HSV.SATURATION: 2}}
                            )})
    assert mask[0, :2].all()
    assert mask[[2,3],:2].all()
    assert not mask[1].any()
    assert not mask[:,2].any()


def test_or_threshold(img_arr):
    mask = apply_threshold(img_arr,
                           {Combine.OR: (
                               {MinMax.MIN: {HSV.HUE: 4}},
                               {MinMax.MAX: {HSV.LIGHTNESS: 6}},
                            )})
    assert not mask[1,0]
    assert mask[[0,2,3]].all()
    assert mask[:,[1,2]].all()


def test_and_or_threshold(img_arr):
    mask = apply_threshold(img_arr,
                           {Combine.AND: (
                               {Combine.OR: (
                                  {MinMax.MAX: {HSV.HUE: 3}},
                                  {MinMax.MIN: {HSV.LIGHTNESS: 5}},
                                )},
                               {MinMax.MAX: {HSV.HUE: 6}}
                            )})
    assert not mask[:, 0].any()
    assert not mask[[0, 2, 3]].any()
    assert mask[1, [1, 2]].all()
