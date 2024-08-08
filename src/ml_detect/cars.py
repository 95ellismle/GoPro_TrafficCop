import cv2 as cv
from datetime import(datetime as datetime_type)
from pathlib import Path
import numpy as np
import pytesseract

from src.data_types import PredictedColor, Image
from src.ml_detect.base import BaseObject, predict
from src.ml_detect.constants import NUM_PLATE_MODEL

from src.cv_detect.colors import (
    segment_colors,
    get_color_segmentations,
    create_color_pallette
)

COUNT = 0
CHARACTERS = []


class NumberPlate(BaseObject):
    # Derived quantities
    _characters: tuple[str] | None = None
    _cleaned_frame: Image | None = None
    _detected_characters: list | None = None
    _number_plate_color: str | None = None
    _number_plate_color_pallette: list[tuple[int, int, int]] | None = None
    _light_color_pallette: list[tuple[int, int, int]] | None = None

    @property
    def characters(self):
        """Extract any characters from the image"""
        if self._characters is not None:
            return self._characters

        img = self.cleaned_frame
        path_to_tess_bin = "/opt/homebrew/bin/tesseract"
        pytesseract.pytesseract.tesseract_cmd = path_to_tess_bin
        return pytesseract.image_to_string(img.img)

    @property
    def light_color_pallette(self):
        """Get the color pallete in the center of the image and remove any very dark colors"""
        global COUNT
        if self._light_color_pallette is not None:
            return self._light_color_pallette

        img = self.frame.copy()

        # Zoom into the center and get the color pallette for it
        shape = img.shape
        shape_d = [int(0.15*i) for i in shape]
        img = img[shape_d[0]:shape[0]-shape_d[0],
                  shape_d[1]:shape[1]-shape_d[1]]

        # Remove dark colors
        img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
        img[img[:, :, 0] < 40] = [0, 128, 128]
        img = cv.cvtColor(img, cv.COLOR_LAB2RGB)

        # Do some color segmentation with Kmedian clustering
        Ncol = 8
        _, colors = get_color_segmentations(img,
                                            Ncol,
                                            do_median=True,
                                            min_clust_size=0.02)

        self._light_color_pallette = colors
        hsv_colors = cv.cvtColor(colors.reshape((1, Ncol, 3)), cv.COLOR_RGB2HSV)
        black = (255 - colors.max(axis=1)) / 255
        for i in range(Ncol):
            if black[i] > 0.9:
                hsv_colors[0][i] = [0,0,0]
        self._light_color_pallette = cv.cvtColor(hsv_colors, cv.COLOR_HSV2RGB)[0]

        return self._light_color_pallette

    @property
    def number_plate_color(self):
        """The number plate is either yellow or white. This function will get which one the number plate is.

        Steps:
            1) Zoom into center
            2) Segment colors to 4
            3) Check the color pallette for any yellow colors, if there are some then set color to yellow. If not then use white.
        """
        if self._number_plate_color is not None:
            return self._number_plate_color

        # Find any yellow colors
        self._number_plate_color = "yellow"
        yellow_color_pallette = self.number_plate_color_pallette
        self._number_plate_color_pallette = None

        if len(yellow_color_pallette) > 0:
            self._number_plate_color = "yellow"
        else:
            self._number_plate_color = "white"

        return self._number_plate_color

    @property
    def number_plate_color_pallette(self):
        """Get the color pallette for picking out the number plate"""
        if self._number_plate_color_pallette is not None:
            return self._number_plate_color_pallette

        colors = self.light_color_pallette[:]

        Ncol = len(colors)
        hsv_colors = cv.cvtColor(colors.reshape((1, Ncol, 3)), cv.COLOR_RGB2HSV)
        black = (255 - colors.max(axis=1)) / 255
        yellow = (1 - (colors[:, 2]/255) - black) / (1 - black)
        yellow[yellow != yellow] = 0.0
        grey = 1 - (np.std(colors, axis=1) / 120.2)

        color_pallette = [[]]
        # Get the most yellow colors
        if self.number_plate_color == "yellow":
            for i in range(Ncol):
                if yellow[i] < 0.4:  # Remove colors that aren't substantially yellow
                    continue
                elif grey[i]  > 0.95:  # Remove any gray colors
                    continue
                # Remove colors with a non-yellow hue
                elif hsv_colors[0][i][0] > 100 or hsv_colors[0][i][1] < 80:
                    continue
                color_pallette[0].append(hsv_colors[0][i])

        # Just get the whitest color
        elif self.number_plate_color == "white":
            color_score = (  (0.3*(1-black))
                           + (0.3*((1-hsv_colors[0,:,1]) / 255))
                           + (0.4*grey)
            )
            ordered = sorted(zip(color_score, hsv_colors[0]),
                            key=lambda i: i[0],
                            reverse=True)
            for score, color in ordered:
                if score > 0.85:
                    color_pallette[0].append(color)

            if len(color_pallette[0]) == 0:
                color_pallette = [[ordered[0][1]]]

        else:
            raise NotImplementedError("Only yellow and white number plates are handled.")

        # Convert from HSV to RGB
        color_pallette = np.array(color_pallette)
        if len(color_pallette[0]) == 0:
            self._number_plate_color_pallette = np.array(color_pallette[0])
            return self._number_plate_color_pallette
        self._number_plate_color_pallette = cv.cvtColor(color_pallette,
                                                        cv.COLOR_HSV2RGB)[0]
        return self._number_plate_color_pallette

    @property
    def cleaned_frame(self):
        """Get colors that are pretty close to the color pallette"""
        global CHARACTERS
        if self._cleaned_frame is not None:
            return self._cleaned_frame

        self.frame = Image(
            cv.bilateralFilter(self.frame.arr,
                               d=9,
                               sigmaColor=50,
                               sigmaSpace=50)
        )

        mask = np.zeros(self.frame.shape, dtype=np.uint8)
        frame_copy = self.frame.copy()
        if self.number_plate_color == 'yellow':
            colors = self.number_plate_color_pallette
            hsv_colors = cv.cvtColor(np.array([colors]), cv.COLOR_RGB2HSV)[0]
            img = cv.cvtColor(self.frame.arr, cv.COLOR_RGB2HSV)
            for color in hsv_colors:
                #dist = np.linalg.norm(self.frame.arr - color, axis=2)
                dist = img - color
                dist = (
                        (0.7  * dist[:, :, 0])
                      + (0.15 * dist[:, :, 1])
                      + (0.05 * dist[:, :, 2])
                )
                mask[dist < 14] = [255, 255, 255]
            gray = 1 - (np.std(frame_copy, axis=2) / 120.2)
            yellow = 1 - (frame_copy[:, :, :2].std(axis=2) / 127.5)
            mask[gray > 0.85] = [0, 0, 0]
            mask[yellow < 0.8] = [0, 0, 0]
            if mask.sum() == 0:
                mask = np.ones_like(self.frame.shape) * 255

        else:
            ...

        if self.number_plate_color == 'yellow':
            global COUNT
            out_dir = Path("/Users/mattellis/Projects/GoProCam/storage/img/num_plate_tidying/out");  out_dir.mkdir(exist_ok=True)
            filepath = out_dir / f"{COUNT}.jpg"
            COUNT += 1

            min_area_rect = cv.minAreaRect(cv.findNonZero(mask[:,:,0]))
            rotated_bounding_box = np.int0(cv.boxPoints(min_area_rect))
            new_mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
            cv.drawContours(new_mask, [rotated_bounding_box], 0, 255, -1)
            new_mask = cv.dilate(new_mask, np.ones((3,3),dtype=np.uint8)).astype(np.bool_)
            final = self.frame.copy()
            final[~new_mask] = [0,0,0]

            angle = -min_area_rect[2]
            if min_area_rect[1][0] < min_area_rect[1][1]:
                angle = 90 + angle
            M = cv.getRotationMatrix2D(min_area_rect[0],-angle, 1)
            final = cv.warpAffine(final, M, self.frame.shape[:2][::-1])

            if COUNT == 2:
                import ipdb; ipdb.set_trace()
                o=1

            clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
            final = cv.cvtColor(final[:], cv.COLOR_RGB2GRAY)
            final = clahe.apply(final)
            final = cv.bilateralFilter(final, d=9, sigmaColor=50, sigmaSpace=50)
            Image(final).save(filepath)
            self._cleaned_frame = Image(final)
        else:
            self._cleaned_frame = self.frame
        return self._cleaned_frame




class Car(BaseObject):
    # Derived quantities
    _number_plate: str | None = None

    # is_moving: bool | None = None
    # color: PredictedColor | None
    # make: str
    # datetime: datetime_type | None = None
    # latitude: float | None = None
    # longitude: float | None = None

    @property
    def number_plate(self) -> str | None:
        """Will get the number plate from an image of a car"""
        if self._number_plate is None:
            self._number_plate = self._get_number_plate()
        return self._number_plate

    def _get_number_plate(self) -> NumberPlate:
        """Find the number plate on the car and return NumberPlate obj"""
        results = predict(self.frame.arr, NUM_PLATE_MODEL)

        number_plates = []
        characters = []
        for result in results:
            for obj_name, boxes in result.items():
                if obj_name == 'number-plates':
                    for box in boxes:
                        number_plate = NumberPlate(self.frame, box)
                        print(number_plate.characters)
                        number_plates.append(number_plate)

#                 elif obj_name == 'character':
#                     for box in boxes:
#                        characters.append(box)

#         # Now add every character to every number plate and let the number plate sort it out
#         for character_box in characters:
#             for number_plate in number_plates:
#                 number_plate.add_character_box(character_box)

