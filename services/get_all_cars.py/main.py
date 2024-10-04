import datetime
from pathlib import Path
import numpy as np

import click
from src.data_types import Image
from src.ml_detect.detect import Detect
from src.go_pro.utils import read_360_video
from src.db import (
    session,
    create_all_tables,
)
from src.db.utils import (
    persist_obj,
    insert_no_conflict,
)
from src.db.vehicles import (
   Directory,
   Video,
   Vehicle,
   Image as ImgTbl,
   Location,
)



class img_writer:
    count = 0
    dir_ = None
    def __init__(self, output_dir):
        self.dir_ = output_dir

    def save_image(self, img: np.ndarray):
        new_path = self.dir_ / f"{self.count}.webp"
        Image(img).save(new_path, "webp")
        self.count += 1
        return new_path


def main(video_file,
         start_frame,
         output_dir):
    create_all_tables()

    video = None
    writer = None
    for frame in read_360_video(video_file, start_frame):
        frame_obs_time = frame.datetime
        if frame.frame_iter == 0:
            output_dir /= f"{video_file.name}-{frame_obs_time:%Y_%m_%d-%H_%M}"
            output_dir.mkdir(exist_ok=True)
            writer = img_writer(output_dir)

            # Create directory and save video file
            video_dir = persist_obj(object_to_insert=Directory(
                                                        dirname=str(video_file.parent)
                                    ),
                                    search_vals=['dirname'])
            img_dir = persist_obj(object_to_insert=Directory(
                                                        dirname=str(output_dir)
                                  ),
                                  search_vals=['dirname'])

            video = persist_obj(Video(filename=str(video_file.name),
                                      directory=video_dir,
                                      observed_at=frame_obs_time),
                                search_vals = ['filename', 'directory'],
            )

        print("\r"f"Frame: {frame.frame_number}       ", end="\r")
        for view in ('front', 'rear',
                     'bottom', 'top',
                     'right', 'left',):
            frame_img = getattr(frame, view)
            cars = Detect(frame_img, 'car')

            for car in cars.extract():
                # Add image to DB
                img_path = writer.save_image(car.frame.arr)
                img_file = persist_obj(
                        object_to_insert=ImgTbl(filename=img_path.name,
                                                directory=img_dir,
                                                observed_at=frame_obs_time,
                                                video=video,
                                                video_time=frame.time),
                        search_vals=['filename', 'directory']
                )

                # Add location to DB
                location = persist_obj(
                        object_to_insert=Location(
                            latitude=frame.metadata['gps']['lat'],
                            longitude=frame.metadata['gps']['lon']
                        ),
                        search_vals=['latitude', 'longitude']
                )

                # Add location to DB
                vehicle = insert_no_conflict(
                        Vehicle,
                        conflict_cols=['video_id', 'video_time', 'vehicle_image_id'],
                        **{'video_id': video.video_id,
                           'vehicle_image_id': img_file.image_id,
                           'location_id': location.location_id,
                           'observed_at': frame_obs_time,
                           'video_time': frame.time},
                )


@click.command()
@click.argument('video_file', type=click.Path(exists=True))
@click.option('--start-frame', type=int, default=0)
@click.option('--output-dir', type=click.Path(exists=True), required=True)
def main_cli(video_file: Path,
             start_frame: int,
             output_dir: Path):
    video_file = Path(video_file)
    output_dir = Path(output_dir)
    main(video_file, start_frame, output_dir)


if __name__ == '__main__':
    main_cli()
