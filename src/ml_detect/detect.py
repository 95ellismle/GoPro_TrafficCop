from typing import Any
from ultralytics.engine.results import Boxes

from src.ml_detect.base import BaseObject, predict
from src.ml_detect.constants import (
    DEVICE,
    YOLOV10_MODEL,
    NUM_PLATE_MODEL,
    MODEL_TYPE,
)
from src.ml_detect.cars import Car, NumberPlate
from src.data_types import Image


class Detect:
    _obj_map: dict[str, tuple[Any, MODEL_TYPE]] = {
        'number-plates':  (NumberPlate, NUM_PLATE_MODEL),
        'character':      (BaseObject, NUM_PLATE_MODEL),
        'car':            (Car,        YOLOV10_MODEL),
        'person':         (BaseObject, YOLOV10_MODEL),
        'bicycle':        (BaseObject, YOLOV10_MODEL),
        'motorcycle':     (BaseObject, YOLOV10_MODEL),
        'airplane':       (BaseObject, YOLOV10_MODEL),
        'bus':            (BaseObject, YOLOV10_MODEL),
        'train':          (BaseObject, YOLOV10_MODEL),
        'truck':          (BaseObject, YOLOV10_MODEL),
        'boat':           (BaseObject, YOLOV10_MODEL),
        'traffic light':  (BaseObject, YOLOV10_MODEL),
        'fire hydrant':   (BaseObject, YOLOV10_MODEL),
        'stop sign':      (BaseObject, YOLOV10_MODEL),
        'parking meter':  (BaseObject, YOLOV10_MODEL),
        'bench':          (BaseObject, YOLOV10_MODEL),
        'bird':           (BaseObject, YOLOV10_MODEL),
        'cat':            (BaseObject, YOLOV10_MODEL),
        'dog':            (BaseObject, YOLOV10_MODEL),
        'horse':          (BaseObject, YOLOV10_MODEL),
        'sheep':          (BaseObject, YOLOV10_MODEL),
        'cow':            (BaseObject, YOLOV10_MODEL),
        'elephant':       (BaseObject, YOLOV10_MODEL),
        'bear':           (BaseObject, YOLOV10_MODEL),
        'zebra':          (BaseObject, YOLOV10_MODEL),
        'giraffe':        (BaseObject, YOLOV10_MODEL),
        'backpack':       (BaseObject, YOLOV10_MODEL),
        'umbrella':       (BaseObject, YOLOV10_MODEL),
        'handbag':        (BaseObject, YOLOV10_MODEL),
        'tie':            (BaseObject, YOLOV10_MODEL),
        'suitcase':       (BaseObject, YOLOV10_MODEL),
        'frisbee':        (BaseObject, YOLOV10_MODEL),
        'skis':           (BaseObject, YOLOV10_MODEL),
        'snowboard':      (BaseObject, YOLOV10_MODEL),
        'sports ball':    (BaseObject, YOLOV10_MODEL),
        'kite':           (BaseObject, YOLOV10_MODEL),
        'baseball bat':   (BaseObject, YOLOV10_MODEL),
        'baseball glove': (BaseObject, YOLOV10_MODEL),
        'skateboard':     (BaseObject, YOLOV10_MODEL),
        'surfboard':      (BaseObject, YOLOV10_MODEL),
        'tennis racket':  (BaseObject, YOLOV10_MODEL),
        'bottle':         (BaseObject, YOLOV10_MODEL),
        'wine glass':     (BaseObject, YOLOV10_MODEL),
        'cup':            (BaseObject, YOLOV10_MODEL),
        'fork':           (BaseObject, YOLOV10_MODEL),
        'knife':          (BaseObject, YOLOV10_MODEL),
        'spoon':          (BaseObject, YOLOV10_MODEL),
        'bowl':           (BaseObject, YOLOV10_MODEL),
        'banana':         (BaseObject, YOLOV10_MODEL),
        'apple':          (BaseObject, YOLOV10_MODEL),
        'sandwich':       (BaseObject, YOLOV10_MODEL),
        'orange':         (BaseObject, YOLOV10_MODEL),
        'broccoli':       (BaseObject, YOLOV10_MODEL),
        'carrot':         (BaseObject, YOLOV10_MODEL),
        'hot dog':        (BaseObject, YOLOV10_MODEL),
        'pizza':          (BaseObject, YOLOV10_MODEL),
        'donut':          (BaseObject, YOLOV10_MODEL),
        'cake':           (BaseObject, YOLOV10_MODEL),
        'chair':          (BaseObject, YOLOV10_MODEL),
        'couch':          (BaseObject, YOLOV10_MODEL),
        'potted plant':   (BaseObject, YOLOV10_MODEL),
        'bed':            (BaseObject, YOLOV10_MODEL),
        'dining table':   (BaseObject, YOLOV10_MODEL),
        'toilet':         (BaseObject, YOLOV10_MODEL),
        'tv':             (BaseObject, YOLOV10_MODEL),
        'laptop':         (BaseObject, YOLOV10_MODEL),
        'mouse':          (BaseObject, YOLOV10_MODEL),
        'remote':         (BaseObject, YOLOV10_MODEL),
        'keyboard':       (BaseObject, YOLOV10_MODEL),
        'cell phone':     (BaseObject, YOLOV10_MODEL),
        'microwave':      (BaseObject, YOLOV10_MODEL),
        'oven':           (BaseObject, YOLOV10_MODEL),
        'toaster':        (BaseObject, YOLOV10_MODEL),
        'sink':           (BaseObject, YOLOV10_MODEL),
        'refrigerator':   (BaseObject, YOLOV10_MODEL),
        'book':           (BaseObject, YOLOV10_MODEL),
        'clock':          (BaseObject, YOLOV10_MODEL),
        'vase':           (BaseObject, YOLOV10_MODEL),
        'scissors':       (BaseObject, YOLOV10_MODEL),
        'teddy bear':     (BaseObject, YOLOV10_MODEL),
        'hair drier':     (BaseObject, YOLOV10_MODEL),
        'toothbrush':     (BaseObject, YOLOV10_MODEL),
    }
    results = None

    def __init__(self, frame: Image, name: str):
        """Factory object for extracting various predictions from images.

        Args:
            frame: The input image
            name: The name of the prediction to extract

        Examples:
            img = Image(filepath)
            cars = Detect(img, 'car')
            for car in cars.extract():
                num_plates = Detect(car.frame, 'number_plate')
        """
        self.frame = frame
        self.name = name
        if name not in self._obj_map:
            joiner = '\n\t* '
            raise ValueError(f"Cannot detect name: {name}. Please choose from:{joiner}{joiner.join(names.keys())}")

        self.ret_obj, self.model = self._obj_map[name]

    def extract(self, conf: float = 0.45,
                name: str | None = None,
                device: str | None = DEVICE):
        """Extract results and yield them

        Args:
            conf: confidence level to use in prediction
            name: defaults to the name given in __init__, though passing in a different name will use the same model to attempt to predcit other objects.
        """
        if name is None: name = self.name

        if self.results is None:
            self.results = predict(self.frame.arr, self.model, conf, device)

        for result in self.results:
            for obj_name, boxes in result.items():
                if name == obj_name:
                    for box in boxes:
                        yield self.ret_obj(self.frame, box[0])
