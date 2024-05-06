from pathlib import Path

import av
import numpy as np
from typing import Iterator

from go_pro.data_types import Img360


def split_360(
        container: av.video.frame.VideoFrame,
        min_time: int | None = None
    ) -> Iterator[Img360]:
    """Split a GoPro max 360 image into front, rear, sides, top and bottom images and return all in a tuple.

    Args:
        img: The 360 image to split
        min_time: Time to seek to before iterating over frames.
                  Unit of time is given in streams.video[0].time_base.
                  On GoPro it seems to be `Fraction(1, 90_000)`

    Returns:
        <np.ndarray, np.ndarray> front, rear image
    """
    if min_time:
        container.seek(min_time)

    stream_fronts = container.streams.video[0]
    stream_rears = container.streams.video[1]

    decoded_stream_fronts = container.decode(stream_fronts)
    decoded_stream_rears = container.decode(stream_rears)

    frame_number = 0
    while True:
        try:
            img_360 = Img360(next(decoded_stream_fronts),
                             next(decoded_stream_rears))
            img_360.frame_number = frame_number
            frame_number += 1
            yield img_360
        except StopIteration:
            break


def read_360_video(filepath: Path, min_time: int) -> Iterator[Img360]:
    """Load a 360 video and return a stream of frames"""
    with av.open(filepath) as container:
        for frame in split_360(container, min_time):
            yield frame
