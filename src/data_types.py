import av
import cv2
import numpy as np
from PIL import Image as PIL_Image
from pathlib import Path

from enum import Enum
from dataclasses import dataclass


tuple_3_uint = tuple[np.uint8, np.uint8, np.uint8]


class Color:
    name: str | None
    rgb: tuple_3_uint

    _color_definitions = {
        "black": (0, 0, 0),
        "bright red": (255, 0, 0),
        "dark red": (100, 0, 0),
        "bright green": (0, 255, 0),
        "dark green": (0, 80, 0),
        "bright blue": (0, 0, 255),
        "navy": (0, 0, 100),
        "yellow": (255, 255, 0),
        "pink": (255, 0, 255),
        "teal": (0, 255, 255),
        "orange": (255, 128, 0),
        "light_gray": (100, 100, 100),
        "dark_gray": (50, 50, 50),
        "white": (200, 200, 200),
    }
    _color_defs_numpy = np.array([np.array(v) for v in _color_definitions.values()])
    _color_defs_keys = list(_color_definitions.keys())

    def __init__(self,
                 rgb: tuple_3_uint | None = None,
                 name: str | None = None):
        if name is not None:
            assert rgb is None, "Please only provide the name of the color or the RGB tuple"
            if name not in self._color_definitions:
                raise ValueError(f"Cannot find color {name} in my definition. Please add it or use one of the following:"
                                 "\n\t* " +  "\n\t* ".join(_color_definitions.keys()))
            rgb = self._color_definitions[name]

        elif rgb is not None:
            assert isinstance(rgb, np.ndarray), "Please input rgb as a 3-tuple numpy array"
            assert rgb.shape==(3,), "Please input the rgb 3-tuple as an np.ndarray of shape (3,)"
            distances = np.linalg.norm(self._color_defs_numpy - np.array(rgb), axis=1)
            name = self._color_defs_keys[np.argmin(distances)]

        else:
            raise ValueError("Please provide either rgb or name of the color")

        self.rgb = rgb
        self.name = name


class PredictedColor(Color):
    certainty: float
    def __init__(self, certainty: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.certainty = certainty


class ColorEnum(Enum):
    RED = 0
    AMBER = 1
    GREEN = 2
    BLACK = 3

class HSV(Enum):
    HUE =  0
    SATURATION = 1
    LIGHTNESS = 2

class MinMax(Enum):
    MIN = 0
    MAX = 1

class Combine(Enum):
    AND = 0
    OR = 1

@dataclass
class Circle:
    center: tuple[float, float]
    radius: float

@dataclass
class Rectangle:
    """pt1 is the point of one corner (in normalised coords).
       pt2 is the point of the opposite corner (normalised coords)
    """
    pt1: tuple[float, float]
    pt2: tuple[float, float]

class Image:
    """Container for image data.

    All operations on either an PIL.Image.Image (e.g: save) should work
    As should all operations on a numpy.ndarray (e.g: T, slicing, mean...)

    We save all data as an Image and a numpy.ndarray for convience -interact with these by interacting with the Image container.

    e.g: Image.arr[:, :688] is equivalent to Image[:, :688]


    Inputs:
        img: PIL Image, numpy array or av VideoFrame.
    """
    arr: np.ndarray
    img: PIL_Image.Image

    def __init__(self,
                 img: PIL_Image.Image|np.ndarray|av.video.frame.VideoFrame|Path|str):
        if isinstance(img, np.ndarray):
            self.arr = img
            self.img = PIL_Image.fromarray(self.arr)
        elif isinstance(img, PIL_Image.Image):
            self.img = img
            self.arr = np.array(self.img)
        elif isinstance(img, av.video.frame.VideoFrame):
            self.img = img.to_image()
            self.arr = np.array(self.img)
        elif isinstance(img, (str, Path)):
            if not Path(img).is_file(): raise FileNotFoundError(f"Can't find file: {img}")
            self.arr = cv2.cvtColor(cv2.imread(str(img)),
                                    cv2.COLOR_BGR2RGB)
            self.img = PIL_Image.fromarray(self.arr)
        else:
            raise TypeError(f"Type {type(img)} not allowed. Please input np.ndarray or PIL.Image.Image")

    def __getitem__(self, *args):
        arr = self.arr.__getitem__(*args)
        return Image(arr)

    def __setitem__(self, *args):
        self.arr = self.arr.__setitem__(*args)
        self.img = Image.fromarray(self.arr)
        return self

    def __getattr__(self, attribute):
        if attribute == 'T':
            ret = np.swapaxes(self, 0, 1)
        else:
            try:
                ret = getattr(self.arr, attribute)
            except AttributeError:
                ret = getattr(self.img, attribute)

        # Return an Image object for things like transpose etc..
        if isinstance(ret, (np.ndarray, PIL_Image.Image)):
            return Image(ret)
        return ret
