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


        if number_plate.number_plate_color == 'white':

            cleaned_frame = number_plate.cleaned_frame.copy()
            bw = cv.adaptiveThreshold(
                                       cv.cvtColor(cleaned_frame, cv.COLOR_RGB2LAB)[:,:,0],
                                       255,
                                       cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY,
                                       15,
                                       -3
            )

            ff = flood_fill_edges(bw)
            ff = remove_small_contours(ff, 0.02)

            kernel = np.ones((2,2), np.uint8)
            ff = cv.dilate(cv.erode(ff, kernel), kernel)
            ff = flood_fill_edges(remove_small_contours(ff, 0.005))

            hsv = cv.cvtColor(cleaned_frame, cv.COLOR_RGB2HSV)

            save_image(
                    np.vstack((
                        cleaned_frame,
                        cv.cvtColor(contrast_boost(hsv[:,:,1]), cv.COLOR_GRAY2RGB),
                        cv.cvtColor(ff, cv.COLOR_GRAY2RGB),
                    )),
                    "white_segments",
                    img_fp.name
            )


        rect = number_plate.bounding_box
        if rect is None:
            continue

        try:
            frame_shape = number_plate.cleaned_frame.shape
            x, y = np.intp(np.array(frame_shape[:2]) / 20)
            rect = ((rect[0][0], rect[0][1]-1), (rect[1][0]+x, rect[1][1]+y), rect[2])
            points = np.intp(cv.boxPoints(rect))
            mask = np.ones_like(number_plate.cleaned_frame[:,:,0])
            cv.drawContours(mask, [points], -1, 0, -1)
            mask = mask.astype(np.bool_)
            fin = number_plate.frame.copy()
            fin[mask] = [0,0,0]

            angle = -rect[2]
            if rect[1][0] < rect[1][1]:
                angle = 90 + angle
            M = cv.getRotationMatrix2D(rect[0],-angle, 1)
            cleaned_frame = cv.warpAffine(fin, M, frame_shape[:2][::-1])
        except:
            print("No number plate found!")

        # Increase brightness and contrast
        cleaned_frame = cv.cvtColor(cleaned_frame, cv.COLOR_RGB2LAB)
        cleaned_frame[:,:,0] = contrast_boost(cleaned_frame[:,:,0])
        cleaned_frame = cv.cvtColor(cleaned_frame, cv.COLOR_LAB2RGB)

        _cleaned_frame = (
                np.vstack((
                    number_plate.frame.arr,
                    cleaned_frame))
        )

        save_image(_cleaned_frame,
                   "segment_colors",
                   img_fp.name)

        del number_plate


@click.command()
@click.option('--image_number', '-in', type=int, default=[-1], multiple=True)
def main_cli(image_number):
    main(image_number)


if __name__ == '__main__':
    main_cli()
