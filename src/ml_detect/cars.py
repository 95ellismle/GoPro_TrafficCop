from ultralytics import YOLO

import cv2
from datetime import(datetime as datetime_type)
from dataclasses import dataclass
import numpy as np

from src.data_types import PredictedColor, Image
from models import MODELS_PATH


MODEL = YOLO(str(MODELS_PATH / 'yolov10l.pt'))


@dataclass
class Car:
    # Inputs
    box: np.ndarray

    # Derived quantities
    car_frame: Image
    number_plate: str | None = None
    is_moving: bool | None = None
    predicted_color: tuple[int, int, int] | None = None
    datetime: datetime_type | None = None
    latitude: float | None = None
    longitude: float | None = None

    def __init__(self, img: Image, box: np.ndarray):
        self.box = box
        self.car_frame = self._calc_car_cutout(img, box)

        self.number_plate = self._calc_number_plate(self.car_frame)
        #self.color = self._calc_color(self.car_frame)
        self.datetime = None
        self.latitude = None
        self.longitude = None

    def _calc_car_cutout(self, img: Image, box: np.ndarray) -> Image | None:
        """Will cutout the car in the image with the specified box"""
        rect = box.xyxy[0].astype(int)
        return img[rect[1]:rect[3], rect[0]:rect[2]]

    def _calc_number_plate(self, car_img: Image) -> str | None:
        """Will get the number plate from an image of a car"""
        return

    def _calc_color(self, car_img: Image) -> tuple[int, int, int] | None:
        """Return color of the car -rgb[255, 255, 255]"""
        num_clusters = 8
        img_arr = car_img.arr
        x, y, _ = img_arr.shape
        certainty = 1
        if x > 120 and y > 120:
            img_arr = img_arr[x//8:7*x//8, y//8:7*y//8]
        else:
            certainty -= 0.2
        if x < 50 or y < 50:
            certainty -= 0.2

        img_arr = cv2.bilateralFilter(img_arr, 15, 50, 50)
        cluster_me = np.float32(img_arr.reshape((-1, 3)))

        criteria = criteria = (cv2.TERM_CRITERIA_EPS ,10 , 1.0)
        ret, label, center = cv2.kmeans(cluster_me,
                                        num_clusters,
                                        None,
                                        criteria,
                                        10,
                                        cv2.KMEANS_PP_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()].reshape((img_arr.shape))
        Image(res).show()

        unique_labels, counts = np.unique(label, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
        second_max_count = sorted(counts)[-2]
        test = second_max_count / counts.max()
        if test > 0.9:
            certainty -= 0.4
        elif test > 0.8:
            certainty -= 0.3
        elif test > 0.7:
            certainty -= 0.2
        elif test > 0.6:
            certainty -= 0.1

        self.color = PredictedColor(rgb=center[most_common_label].astype(np.uint8), certainty=certainty)
        print(f"Shitty {self.color.name} car (rgb: {self.color.rgb}), certainty: {self.color.certainty}")
        return


class Detect:
    _obj_map = {'car': Car}

    def __init__(self, frame: Image):
        self.results = MODEL.predict(source=frame.arr)
        self.frame = frame

    def extract(self, name: str):
        for result in self.results:
            names = {v: k for k, v in result.names.items()}
            if name not in names or name not in self._obj_map:
                joiner = '\n\t* '
                raise ValueError(f"Cannot detect name: {name}. Please choose from:{joiner}{joiner.join(names.keys())}")

            for ibox, box in enumerate(result.boxes.numpy()):
                if box.cls == names[name]:
                    yield self._obj_map[name](self.frame, box)
