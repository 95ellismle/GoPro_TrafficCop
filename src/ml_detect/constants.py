from ultralytics import YOLO
from ultralytics import YOLOv10
from ultralytics.models.yolov10.model import YOLOv10 as yolo_v10_model
from ultralytics.models.yolo.model import YOLO as yolo_model

from models import MODELS_PATH

MODEL_TYPE = yolo_v10_model | yolo_model

YOLOV10_MODEL = YOLOv10(str(MODELS_PATH / 'yolov10l.pt'))
NUM_PLATE_MODEL = YOLOv10(str(MODELS_PATH / 'num_plates_with_gopro_img.pt'))

DEVICE: str | None = "mps"
