from pathlib import Path

import click

from src.ml_detect.cars import Detect
from src.go_pro.utils import read_360_video

from storage import STORAGE_PATH

DATA_DIR = STORAGE_PATH / "img/training/cars"
if not DATA_DIR.is_dir():
    DATA_DIR.mkdir(parents=True, exists_ok=True)


def main(video_file, start_frame):
    for frame in read_360_video(video_file, start_frame):
        print("\r"f"Frame: {frame.frame_number}       ", end="\r")
        for view in ('front', 'rear',
                      'front_bottom', 'front_top',
                      'rear_bottom', 'rear_top',
                      'rear_right', 'rear_left',
                      'front_right', 'front_left',):

            frame_img = getattr(frame, view)
            extractor = Detect(frame_img)

            all_hash_vals = {int(i.stem.split('_')[0])
                             for i in DATA_DIR.glob('*_car_*.jpg')}

            count = 0
            for car in extractor.extract('car'):
                img_shape = car.car_frame.arr.shape
                if img_shape[0] < 200 or img_shape[1] < 200:
                    continue

                hash_val = str(car.car_frame.arr.mean()).replace('.', '')
                if hash_val in all_hash_vals:
                    continue
                filepath = DATA_DIR / f"{hash_val}_car.jpg"
                while filepath.exists():
                    count += 1
                    filepath = DATA_DIR / f"{hash_val}_car_{count}.jpg"

                car.car_frame.img.save(filepath)



@click.command()
@click.argument('video_file', type=click.Path(exists=True))
@click.option('--start_frame', type=int, default=0)
def main_cli(video_file: Path, start_frame: int):
    """Analyse a video file and pick out cars that skip red lights"""
    video_file = Path(video_file)
    main(video_file, start_frame)


if __name__ == '__main__':
    main_cli()
