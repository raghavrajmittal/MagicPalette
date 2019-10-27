# python yolo_real_time.py
# python yolo_real_time.py --output output/airport.avi (for saving the live video to a file)

from webcam import Webcam
import numpy as np
import argparse
import time
import cv2
import os

MIN_CONFIDENCE = 0.	# minimum confidence needed when filtering out detections
THRESHOLD = 0.3			# non-maxima suppression threshold
LABELS = open("yolo-coco/coco.names").read().strip().split("\n") # load COCO class labels that YOLO model was trained on
ONLY_SHOW = ["toothbrush"]
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")
ln = net.getLayerNames() # determine *output* layer names of the YOLO object detector
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# in case video needs to be saved
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default=False,
	help="path to output video")
args = vars(ap.parse_args())
if args["output"] is not False:
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	writer = cv2.VideoWriter(args["output"], fourcc, 2,(640, 360), True)

print("[INFO] Starting webcam...")
webcam = Webcam()
webcam.start()
H, W = None, None
canvas = None
previous_point = None

print("[INFO] You're good to draw!")
while True:
	# get current frame and image dimensions
	frame = webcam.get_current_frame()
	frame = cv2.flip(frame, 1)
	frame = cv2.resize(frame, (640, 360))
	if H is None:
		(H, W) = frame.shape[:2]
		canvas = 255 * np.ones(shape=[H, W, 3], dtype=np.uint8)


	# construct a blob from the input frame and then perform a forward pass of the YOLO object detector
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)


	boxes = []
	confidences = []
	classIDs = []
	# get all confident detections and bounding boxes
	for output in layerOutputs: 	# loop over each of the layer outputs
		for detection in output:		# loop over each of the detections
			# extract the class ID and confidence (i.e., probability) of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			if LABELS[classID] not in ONLY_SHOW:
				continue
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected probability is greater than the minimum probability
			if confidence > MIN_CONFIDENCE:
				# scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# update our list of bounding box coordinates, confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)


	# apply non-maxima suppression to suppress weak, overlapping bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, THRESHOLD)


	if len(idxs) > 0:
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw on canvas
			# canvas_startY = int(y + (0.1 * (h)))
			# canvas_endY = int(y + (0.12 * (h)))
			# canvas_startX = int(x + (0.45 * (w)))
			# canvas_endX = int(x + (0.55 * (w)))
			# canvas[canvas_startY: canvas_endY, canvas_startX: canvas_endX] = (0, 0, 255)
			canvas_y = int(y + (0.1 * (h)))
			canvas_x = int(x + (0.5 * (w)))
			if previous_point is not None:
				cv2.line(canvas,(canvas_x, canvas_y), previous_point,(0,0,255),10)
			previous_point = canvas_x, canvas_y

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


	if args["output"] is not False:
		writer.write(frame)

	cv2.imshow("Output", frame)
	cv2.imshow("Canvas", canvas)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


print("[INFO] cleanup up...")
if args["output"] is not False:
	writer.release()
webcam.end()
