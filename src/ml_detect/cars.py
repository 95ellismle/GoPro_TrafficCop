import cv2 as cv
from datetime import(datetime as datetime_type)
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterator

from src.data_types import PredictedColor, Image
from src.ml_detect.base import BaseObject, predict
from src.ml_detect.constants import NUM_PLATE_MODEL, CHARACTER_MODEL

from src.cv_detect.colors import (
    segment_colors,
    get_color_segmentations,
    create_color_pallette
)
from src.cv_detect.common import (
    contrast_boost,
    join_clean_mask,
    remove_small_contours,
    flood_fill_edges,
    snip_little_lines,
    derivative_snipping,
)


COUNT = 0


def create_filepath(filepath, new_dir):
    dir_ = filepath.parent / new_dir
    dir_.mkdir(exist_ok=True)
    return dir_ / filepath.name


class NumberPlate(BaseObject):
    """
    Contains utils to tidy images before passing them through a ML model to detect characters.

    The processes that should take place are:
        0) Identify if the num plate is yellow or white
        1) The actual number plate should be identified, this is different for yellow and white num plates
        2) Rotate image so the long axis of the number plate is aligned with the horizontal axis
        3) Smooth image
    """
    # Derived quantities
    _characters: str | None = None
    _cleaned_frame: Image | None = None
    _detected_characters: list | None = None
    _number_plate_color: str | None = None
    _number_plate_color_pallette: list[tuple[int, int, int]] | None = None
    _light_color_pallette: list[tuple[int, int, int]] | None = None
    _bounding_box: np.ndarray | None = None
    _final_blobs: np.ndarray | None = None
    _rotation: float | None = None
    _rotated_frame: np.ndarray | None = None
    _final_frame: np.ndarray | None = None

    @property
    def characters(self):
        """Extract any characters from the image"""
        if self._characters is not None:
            return self._characters

        img = self.final_frame
        results = predict(img, CHARACTER_MODEL, conf=0.2)

        # Sort chars so they read from left to right
        results = sorted([(v[0].xywh[0,0], k)
                          for k,v in results[0].items()])
        self._characters = ''.join([i[1] for i in results])
        return self._characters

    @property
    def light_color_pallette(self):
        """Get the color pallete in the center of the image and remove any very dark colors"""
        if self._light_color_pallette is not None:
            return self._light_color_pallette

        img = self.cleaned_frame.copy()

        # Zoom into the center and get the color pallette for it
        shape = img.shape
        shape_d = [int(0.3*shape[0]), int(0.15*shape[0])]
        img = img[shape_d[0]:shape[0]-shape_d[0],
                  shape_d[1]:shape[1]-shape_d[1]]

        # Remove dark colors
        img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
        img[contrast_boost(img[:, :, 0]) < 80] = [0, 128, 128]
        img = cv.cvtColor(img, cv.COLOR_LAB2RGB)

        # Do some color segmentation with Kmedian clustering
        Ncol = 8
        _, colors = get_color_segmentations(img,
                                            Ncol,
                                            do_median=True,
                                            min_clust_size=0.05)

        self._light_color_pallette = colors
        hsv_colors = cv.cvtColor(colors.reshape((1, Ncol, 3)), cv.COLOR_RGB2HSV)
        black = (255 - colors.max(axis=1)) / 255
        for i in range(Ncol):
            if black[i] > 0.9:
                hsv_colors[0][i] = [0,0,0]
        self._light_color_pallette = np.array(
            [
                i for i in cv.cvtColor(hsv_colors, cv.COLOR_HSV2RGB)[0]
                if set(i) != {0}
            ]
        )
        # if len(self._light_color_pallette) == 0:
        #     import ipdb; ipdb.set_trace()

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
        yellow = np.zeros_like(black)
        mask = black < 0.999
        yellow[mask] = (1 - (colors[mask, 2]/255) - black[mask]) / (1 - black[mask])
        grey = 1 - (np.std(colors, axis=1) / 120.2)

        color_pallette = [[]]
        # Get the most yellow colors
        if self.number_plate_color == "yellow":
            for i in range(Ncol):
                hue, sat, val = hsv_colors[0][i]
                if yellow[i] < 0.3:  # Remove colors that aren't substantially yellow
                    continue
                elif grey[i]  > 0.91:  # Remove any gray colors
                    continue
                # Remove colors with a non-yellow hue
                # Yellows are hues between 9->39 & 192->212
                elif hue < 9 or hue > 212:
                    continue
                elif hue > 39 and hue < 192:
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

    def _get_white_final_blobs(self):
        """Get the final blobs that show roughly where the yellow number plate is"""
        cleaned_frame = self.cleaned_frame.copy()
        final = cv.adaptiveThreshold(
                                   cv.cvtColor(cleaned_frame, cv.COLOR_RGB2LAB)[:,:,0],
                                   255,
                                   cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY,
                                   15,
                                   -3
        )

        final = flood_fill_edges(final)
        final = remove_small_contours(final, 0.02)

        final = snip_little_lines(final, (2, 2))
        final = derivative_snipping(final)
        final = remove_small_contours(final, 0.005)
        final = flood_fill_edges(final, max_change=0.8)
        return final

    def _get_yellow_final_blobs(self):
        """Get the final blobs that show roughly where the yellow number plate is"""
        cleaned_frame = self.cleaned_frame.copy()

        hsv = cv.cvtColor(cleaned_frame, cv.COLOR_RGB2HSV)
        yellow_hue = hsv.copy()
        yellow_hue[(hsv[:,:,0] < 13) | (hsv[:,:,0] > 27)] = [0,0,0]
        yellow_hue = cv.cvtColor(yellow_hue, cv.COLOR_HSV2RGB)

        grey = cv.adaptiveThreshold(
                contrast_boost(np.std(cleaned_frame.copy(), axis=2)),
                255,
                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY,
                15,
                -3)

        # Black out red and blue
        grey[(hsv[:,:,0] < 7) | ((hsv[:,:,0] > 92) & (hsv[:,:,0] < 110))] = 0
        grey = flood_fill_edges(grey)
        grey = remove_small_contours(remove_small_contours(grey, threshold=0.025), threshold=1e-3)

        grey = snip_little_lines(grey)
        grey = remove_small_contours(grey, 1e-3)

        final = yellow_hue.copy()
        final[grey == 0] = 0
        return cv.cvtColor(final, cv.COLOR_RGB2GRAY)

    @property
    def final_blobs(self):
        """Will get the mask that defines the number plate, is not perfect though if a bounding box is
        drawn around it, it mostly find the number plate.
        """
        if self._final_blobs is not None:
            return self._final_blobs

        if self.number_plate_color == 'yellow':
            self._final_blobs = self._get_yellow_final_blobs()
        elif self.number_plate_color == 'white':
            self._final_blobs = self._get_white_final_blobs()
        return self._final_blobs

    @property
    def bounding_box(self):
        """Will get the rotated rectangle representing the minimum area bounding box around the num plate"""
        if self._bounding_box is not None:
            return self._bounding_box

        final = self.final_blobs
        try:
            self._bounding_box = cv.minAreaRect(cv.findNonZero(final, cv.COLOR_RGB2GRAY))
        except:
            return None

        return self._bounding_box

    @property
    def cleaned_frame(self):
        """We tidy the frame image and try to increase the contrast between background and number plate characters.

        This is slightly different for white and yellow images, but they both do the following:
            1) Identify the number plate background by using colors &/or lightness gradients etc..
            2) Remove background objects by blacking them out
            3) Find any characters and reduce their lightness
            4) Find any background and increase its lightness
            5) Find the min area bounding box around the number plate and rotate so it's long axis is parallel with the x axis
        """
        if self._cleaned_frame is not None:
            return self._cleaned_frame

        cleaned_frame = cv.bilateralFilter(self.frame.arr, d=21, sigmaColor=15, sigmaSpace=15)
        cleaned_frame = segment_colors(cleaned_frame, 11)

        self._cleaned_frame = Image(cleaned_frame)

        return self._cleaned_frame

    @property
    def rotation(self):
        """Get the rotation of the number plate such that the long axis of the number plate aligns with the horizontal
        axis of the image.
        """
        if self._rotation is not None:
            return self._rotation

        try:
            _, _, self._rotation = cv.fitEllipse(cv.findNonZero(self.final_blobs))
        except:
            self._rotation = 90
        self._rotation = 90 - self._rotation
        return self._rotation

    @property
    def rotated_frame(self):
        """Return the cleaned rotated number plate so the long axis of the number plate aligns with the horizontal axis
        of the image.
        """
        global COUNT
        if self._rotated_frame is not None:
            return self._rotated_frame

        rotation = self.rotation
        rect = self.bounding_box
        if rect is None:
            self._rotated_frame = self.cleaned_frame
            return self._rotated_frame

        frame_shape = self.frame.shape

        x, y = np.array(frame_shape[:2]) / 15
        rect = ((rect[0][0], rect[0][1]), (rect[1][0]+x, rect[1][1]+y), rect[2])
        points = np.intp(cv.boxPoints(rect))
        mask = np.ones_like(self.frame[:,:,0])
        cv.drawContours(mask, [points], -1, 0, -1)
        mask = mask.astype(np.bool_)
        fin = self.frame.copy()
        fin[mask] = [0,0,0]


        M = cv.getRotationMatrix2D(rect[0],-rotation, 1)
        self._rotated_frame = cv.warpAffine(fin, M, frame_shape[:2][::-1])
        return self._rotated_frame

    @property
    def final_frame(self):
        """Will return the final frame: cleaned, rotated and cutout."""
        if self._final_frame is not None:
            return self._final_frame

        new_img = cv.bilateralFilter(self.rotated_frame.copy(),
                                     d=21,
                                     sigmaSpace=11,
                                     sigmaColor=11)
        new_img = cv.cvtColor(new_img, cv.COLOR_RGB2LAB)
        new_img[:,:,0] = contrast_boost(new_img[:,:,0])
        new_img = cv.cvtColor(new_img, cv.COLOR_LAB2RGB)

        sharpen_kern = np.array([[0, -0.5/4, 0],
                                 [-0.5/4, 1.5, -0.5/4],
                                 [0, -0.5/4, 0]])
        new_img = cv.filter2D(new_img, ddepth=-1, kernel=sharpen_kern)
        new_img = cv.bilateralFilter(new_img,
                                     d=5,
                                     sigmaSpace=17,
                                     sigmaColor=17)
        self._final_frame = new_img

        return self._final_frame


class Car(BaseObject):
    # Derived quantities
    _number_plates: str | None = None

    # is_moving: bool | None = None
    # color: PredictedColor | None
    # make: str
    # datetime: datetime_type | None = None
    # latitude: float | None = None
    # longitude: float | None = None

    @property
    def number_plates(self) -> str | None:
        """Will get the number plate from an image of a car"""
        if self._number_plates is None:
            self._number_plates = self._get_number_plates()
        return self._number_plates

    def _get_number_plates(self) -> Iterator[NumberPlate]:
        """Find the number plate on the car and return NumberPlate obj"""
        results = predict(self.frame.arr, NUM_PLATE_MODEL)

        number_plates = []
        for result in results:
            for obj_name, boxes in result.items():
                if obj_name == 'number-plates':
                    for box in boxes:
                        w, h = box.xywh[0, [2,3]] // 20
                        box.xyxy[0, 0] -= w
                        box.xyxy[0, 2] += w
                        box.xyxy[0, 1] -= h
                        box.xyxy[0, 3] += h
                        if box.xyxy[0,0] < 0: box.xyxy[0,0] = 0
                        if box.xyxy[0,1] < 0: box.xyxy[0,1] = 0
                        number_plate = NumberPlate(self.frame, box)
                        number_plates.append(number_plate)
        return number_plates
