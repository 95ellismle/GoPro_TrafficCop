from ultralytics import YOLO  # This takes quite a while
from ultralytics import YOLOv10
from ultralytics.models.yolov10.model import YOLOv10 as yolo_v10_model
from ultralytics.models.yolo.model import YOLO as yolo_model

from models import MODELS_PATH

MODEL_TYPE = yolo_v10_model | yolo_model
DEVICE: str | None = "mps"

YOLOV10_MODEL = YOLOv10(str(MODELS_PATH / 'yolov10l.pt'))
NUM_PLATE_MODEL = YOLOv10(str(MODELS_PATH / 'num_plates_with_gopro_img.pt'))
CHARACTER_MODEL = YOLOv10(str(MODELS_PATH / 'characters.pt'))

