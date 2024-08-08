yolo \
	model=yolov10l.pt \
	patience=10 \
	device=mps \
	task=detect \
	mode=train \
	epochs=100 \
	batch=6 \
	plots=True \
	name=num_plates \
	model=/Users/mattellis/Projects/GoProCam/models/yolov10l.pt \
	data=/Users/mattellis/Projects/GoProCam/storage/img/training/num_plates/data.yaml
