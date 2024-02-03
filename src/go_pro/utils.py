import av
import numpy as np
from typing import Iterator

from go_pro.data_types import Img360


def split_360(
        container: av.video.frame.VideoFrame
    ) -> Iterator[Img360]:
    """Split a GoPro max 360 image into front, rear, sides, top and bottom images and return all in a tuple.

    Args:
        img: The 360 image to split

    Returns:
        <np.ndarray, np.ndarray> front, rear image
    """
    stream_fronts = container.streams.video[0]
    stream_rears = container.streams.video[1]

    decoded_stream_fronts = container.decode(stream_fronts)
    decoded_stream_rears = container.decode(stream_rears)

    while True:
        try:
            yield Img360(next(decoded_stream_fronts),
                         next(decoded_stream_rears))
        except StopIteration:
            break


