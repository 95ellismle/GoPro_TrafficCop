import numpy as np

from src.data_types import PredictedColor, Image
from src.ml_detect.constants import (
    DEVICE,
    MODEL_TYPE,
)


class BaseObject:
    # Inputs
    box: np.ndarray

    # cutout from original image
    frame: Image

    def __init__(self, img: Image, box: np.ndarray):
        self.box = box
        self.frame = self._calc_cutout(img, box)

    def _calc_cutout(self, img: Image, box: np.ndarray) -> Image | None:
        """Will cutout the car in the image with the specified box"""
        rect = box.xyxy[0].astype(int)
        return img[rect[1]:rect[3], rect[0]:rect[2]]


def predict(img: Image,
            model: MODEL_TYPE,
            conf: float = 0.4,
            device: str | None = DEVICE):
    """Actually run a model on an image and return whatever is found as a dict"""
    if device == "mps":
        results = [i.cpu() for i in
                   model.predict(img, conf=conf, device=device)]
    else:
        results = mode.predict(img, conf=conf)

    all_res = []
    for result in results:
        boxes = {}
        for ibox, box in enumerate(result.boxes.numpy()):
            name = result.names[int(box.cls[0])]
            boxes.setdefault(name, []).append(box)
        all_res.append(boxes)
    return all_res


