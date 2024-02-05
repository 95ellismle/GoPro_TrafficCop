import av
import cv2
import numpy as np
from PIL import Image as PIL_Image
from pathlib import Path


class Image:
    """Container for image data.

    All operations on either an PIL.Image.Image (e.g: save) should work
    As should all operations on a numpy.ndarray (e.g: T, slicing, mean...)

    We save all data as an Image and a numpy.ndarray for convience -interact with these by interacting with the Img container.

    e.g: Img.arr[:, :688] is equivalent to Img[:, :688]


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
            self.arr = cv2.cvtColor(cv2.imread(str(img)),
                                    cv2.COLOR_BGR2RGB)
            self.img = PIL_Image.fromarray(self.arr)
        else:
            raise TypeError(f"Type {type(img)} not allowed. Please input np.ndarray or PIL.Image.Image")

    def __getitem__(self, *args):
        arr = self.arr.__getitem__(*args)
        return Img(arr)

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
            return Img(ret)
        return ret
