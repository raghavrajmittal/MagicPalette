# python real_time_object_detection.py

import numpy as np
import time
import cv2
from webcam import Webcam

MIN_CONFIDENCE = 0.1	# minimum probability to filter weak detections
prototext_path = "mobileNetSSD/MobileNetSSD_deploy.prototxt.txt" 	# path to Caffe 'deploy' prototxt file
model_path = "mobileNetSSD/MobileNetSSD_deploy.caffemodel"			# path to Caffe pre-trained model

# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
ONLY_SHOW = ["bottle"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe(prototext_path, model_path)	# load our serialized model from disk
webcam = Webcam()
webcam.start()

h, w = None, None
canvas = None
previous_point = None

while True:
	frame = webcam.get_current_frame()
	frame = cv2.resize(frame, (640, 360))
	if h is None:
		(h, w) = frame.shape[:2]
		canvas = 255 * np.ones(shape=[h, w, 3], dtype=np.uint8)


	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated wih the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > MIN_CONFIDENCE:
			# extract the index of the class label from the detections`,
			# then compute the (x, y)-coordinates of the bounding box for the object
			idx = int(detections[0, 0, i, 1])

			if CLASSES[idx] not in ONLY_SHOW:
				continue


			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw on canvas
			# canvas_startY = int(startY + (0.1 * (endY - startY)))
			# canvas_endY = int(startY + (0.12 * (endY - startY)))
			# canvas_startX = int(startX + (0.45 * (endX - startX)))
			# canvas_endX = int(startX + (0.55 * (endX - startX)))
			# canvas[canvas_startY: canvas_endY, canvas_startX: canvas_endX] = (0, 0, 255)
			canvas_y = int(startY + (0.1 * (endY - startY)))
			canvas_x = int(startX + (0.5 * (endX - startX)))
			if previous_point is not None:
				cv2.line(canvas,(canvas_y, canvas_x), previous_point,(0,0,255),15)
			previous_point = canvas_y, canvas_x

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

			# for making a collage of labels
			# cv2.putText(canvas, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			# cv2.rectangle(canvas, (startX, startY), (endX, endY), COLORS[idx], 2)

	cv2.imshow("frame", frame)
	cv2.imshow("canvas", canvas)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

# cleanup
webcam.end()
