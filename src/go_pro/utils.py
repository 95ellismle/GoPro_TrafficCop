from pathlib import Path
import datetime

import gpmf
import av
import numpy as np
from typing import Iterator

from go_pro.data_types import Img360_Cube


FRACS = [np.array([datetime.timedelta(seconds=i/j) for i in range(j)])
         for j in range(1, 31)]


def split_360(
        container: av.video.frame.VideoFrame,
        min_time: int | None = None,
        metadata: list | None = None,
    ) -> Iterator[Img360_Cube]:
    """Split a GoPro max 360 image into front, rear, sides, top and bottom images and return all in a tuple.

    Args:
        img: The 360 image to split
        min_time: Time to seek to before iterating over frames.
                  Unit of time is given in streams.video[0].time_base.
                  On GoPro it seems to be `1 / 90_000`

    Returns:
        <np.ndarray, np.ndarray> front, rear image
    """
    if min_time:
        container.seek(min_time)

    stream_fronts = container.streams.video[0]
    stream_rears = container.streams.video[1]
    frame_rate = int(stream_fronts.average_rate)

    decoded_stream_fronts = container.decode(stream_fronts)
    decoded_stream_rears = container.decode(stream_rears)

    frame_number = 0
    while True:
        try:
            dsf = next(decoded_stream_fronts)
            dsr = next(decoded_stream_rears)
            img_360 = Img360_Cube(dsf, dsr)
            img_360.time = dsf.time
            img_360.metadata = get_metadata_for_time(metadata, img_360.time)
            img_360.datetime = img_360.metadata['gps']['datetime']
            img_360.frame_number = int(frame_rate * img_360.time)
            img_360.frame_iter = frame_number
            frame_number += 1
            yield img_360
        except StopIteration:
            break


def _indiv_metadata_selector(metadata: dict, search_time: float):
    """Use optimised search to find the correct metadata for 1 category.

    This scales as O(1)."""
    assert 'time' in metadata.keys(), f"Cannot find time array in metadata"
    times = metadata['time']
    approx_diff = approx_diff = (times[-1] - times[0]) / len(times)
    approx_ind = int(search_time / approx_diff)

    # Get a much smaller array -where the search time is found
    gap = 50
    min_ind = max(0, approx_ind - gap)
    max_ind = approx_ind + gap
    search_arr = times[min_ind: max_ind]

    # Use bin search in new much smaller array
    ind = np.searchsorted(search_arr, search_time) + min_ind
    return {k: v[ind] for k, v in metadata.items()}


def get_metadata_for_time(metadata: dict[str, dict], time: float):
    """Use optimised search to find the correct metadata for each category"""
    this_metadata = {k: _indiv_metadata_selector(v, time)
                     for k, v in metadata.items()}
    return this_metadata


def parse_gps_metadata(metadata):
    """Parses the GPS metadata into a more useful format, specifically:

    {'time': [],
     'datetime': [],
     'lat': [],
     ...}

    The arrays are all sorted wrt time for very efficient lookup.
    """
    gps_data = {'time': [],
                'datetime': [],
                'lat': [],
                'lon': [],
                'alt': [],
                'vel_2d': [],
                'vel_3d': []}
    first_timestamp = None
    len_meta = len(metadata)-1
    for imeta, sec in enumerate(metadata):
        meta = sec[2][10][2]

        timestamp = datetime.datetime.strptime(meta[4].value, "%Y-%m-%d %H:%M:%S.%f")
        timestamp = timestamp.astimezone(datetime.timezone.utc)
        data = meta[9].value / meta[7].value

        if first_timestamp is None:
            first_timestamp = timestamp
        time = timestamp - first_timestamp
        len_ = len(data)

        fracs = FRACS[len_]
        if imeta == len_meta:
            fracs = FRACS[18]

        gps_data['time'].extend((time+i).total_seconds() for i in fracs[:len_])
        gps_data['datetime'].extend(timestamp + fracs[:len_])
        gps_data['lat'].extend(data[:,0])
        gps_data['lon'].extend(data[:,1])
        gps_data['alt'].extend(data[:,2])
        gps_data['vel_2d'].extend(data[:,3])
        gps_data['vel_3d'].extend(data[:,4])

    # Quick validation
    assert len(gps_data['time']) == len(gps_data['lat']), f"Time and latitude have differing lens, please check why"
    # Now sort the arrays
    if sorted(gps_data['time']) != gps_data['time']:
        inds = sorted(range(len(gps_data['time'])), key=gps_data['time'].__getitem__)
        for name in ('time', 'datetime', 'lat', 'lon', 'alt', 'vel_2d', 'vel_3d'):
            gps_data[name] = [gps_data[name][i] for i in inds]

    return {'gps': gps_data}


def read_360_video(filepath: Path,
                   min_time: int | None = None,
                   read_metadata: bool = True) -> Iterator[Img360_Cube]:
    """Load a 360 video and return a stream of frames"""
    with av.open(filepath) as container:
        # Read metadata if required
        metadata = None
        if read_metadata:
            stream = gpmf.io.extract_gpmf_stream(str(filepath))
            metadata = gpmf.parse.expand_klv(stream)
            parsed_metadata = {
                **parse_gps_metadata(metadata)
            }

        for frame in split_360(container,
                               min_time,
                               parsed_metadata):
            yield frame
