from pathlib import Path
import numpy as np

import click
from src.data_types import Image
from src.ml_detect.detect import Detect
from src.go_pro.utils import read_360_video

from storage import STORAGE_PATH

COUNT = 0
DATA_DIR = STORAGE_PATH / "img/bare_number_plates"
if not DATA_DIR.is_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_image(img: np.ndarray, dir_name: str):
    new_dir = DATA_DIR / dir_name
    new_dir.mkdir(exist_ok=True)
    new_path = new_dir / f"{COUNT}.jpg"
    Image(img).save(new_path)


def main(video_file, start_frame):
    global COUNT

    for frame in read_360_video(video_file, start_frame):
        print("\r"f"Frame: {frame.frame_number}       ", end="\r")
        for view in ('front',# 'rear',
                      #'front_bottom', 'front_top',
                      #'rear_bottom', 'rear_top',
                      #'rear_right', 'rear_left',
                      #'front_right', 'front_left',
                     ):

            frame_img = getattr(frame, view)
            cars = Detect(frame_img, 'car')

            all_hash_vals = {int(i.stem.split('_')[0])
                             for i in DATA_DIR.glob('*_car_*.jpg')}

            count = 0
            for car in cars.extract():
                img_shape = car.frame.arr.shape
                if img_shape[0] < 200 or img_shape[1] < 200:
                    continue

                for number_plate in car.number_plates:
                    COUNT += 1

                    save_image(number_plate.frame.arr, "original")


@click.command()
@click.argument('video_file', type=click.Path(exists=True))
@click.option('--start_frame', type=int, default=0)
def main_cli(video_file: Path, start_frame: int):
    video_file = Path(video_file)
    main(video_file, start_frame)


if __name__ == '__main__':
    main_cli()
