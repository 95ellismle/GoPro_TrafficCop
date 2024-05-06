from src.form_fill.main import FormFill

import pytest


URL = "https://www.met.police.uk/ro/report/rti/rti-beta-2.1/report-a-road-traffic-incident/"


def test_add_location():
    lat = 51.56828783010922
    lon = -0.12375579668981951
    with FormFill(url=URL) as ff:
        ff.add_location(lat, lon)

