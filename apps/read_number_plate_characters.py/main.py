from pathlib import Path
import numpy as np
import click
import cv2 as cv

from src.data_types import Image
from src.ml_detect.cars import NumberPlate
from src.cv_detect.common import (
    contrast_boost,
    remove_small_contours,
    flood_fill_edges,
    snip_little_lines,
    rectangliness,
    count_islands,
    derivative_snipping,
)

from storage import STORAGE_PATH

IN_DIR = STORAGE_PATH / "img/bare_number_plates/original"
OUT_DIR = STORAGE_PATH / "img/number_plate_tidying"
if not OUT_DIR.is_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_image(img: np.ndarray, dir_name: str, filename: str):
    new_dir = OUT_DIR / dir_name
    new_dir.mkdir(exist_ok=True)
    new_path = new_dir / filename
    Image(img).save(new_path)


def main(image_numbers):
    if image_numbers == (-1,):
        images = IN_DIR.glob('*.jpg')
    else:
        images = (IN_DIR / f'{i}.jpg' for i in image_numbers)
        images = (i for i in images if i.is_file())
    images = sorted([i for i in images], key=lambda i: int(i.stem))

    count = 0
    for img_fp in images:
        number_plate = NumberPlate(Image(img_fp))
        print(f"{count}/{len(images)}")
        count += 1

        save_image(number_plate.final_frame,
                   "segment_colors",
                   img_fp.name)

        del number_plate


@click.command()
@click.option('--image_number', '-in', type=int, default=[-1], multiple=True)
def main_cli(image_number):
    main(image_number)


if __name__ == '__main__':
    main_cli()
