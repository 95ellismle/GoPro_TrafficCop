from ultralytics import YOLO  # This takes quite a while
#from ultralytics import YOLOv10
#from ultralytics.models.yolov10.model import YOLOv10 as yolo_v10_model
from ultralytics.models.yolo.model import YOLO as yolo_model

from models import MODELS_PATH


MODEL_TYPE = yolo_model
# MODEL_TYPE = yolo_v10_model | yolo_model

YOLOV10_MODEL = YOLO(str(MODELS_PATH / 'yolo11m.pt'))  # yolov10l.pt'))
NUM_PLATE_MODEL = YOLO(str(MODELS_PATH / 'num_plates_with_gopro_img.pt'))
CHARACTER_MODEL = YOLO(str(MODELS_PATH / 'characters.pt'))

