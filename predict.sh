yolo \
	predict \
	model=models/num_plates_with_gopro_img.pt \
	conf=0.45 \
	device=mps \
	source="storage/training/num_plates/*/images/*.jpg"
