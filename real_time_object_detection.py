# python real_time_object_detection.py
import numpy as np
import time
import cv2
from webcam import Webcam
from keras.preprocessing import image
from keras.models import model_from_json, load_model
from playsound import playsound


# find most frequest expression in a given list
def mostFrequent(expressionArr):
	n = len(expressionArr)
	arr = sorted(expressionArr)
	max_count = 1; res = arr[0]; curr_count = 1

	for i in range(1, n):
		if (arr[i] == arr[i - 1]):
			curr_count += 1
		else:
			if (curr_count > max_count):
				max_count = curr_count
				res = arr[i - 1]
			curr_count = 1

	if (curr_count > max_count):
		max_count = curr_count
		res = arr[n - 1]
	return res


MIN_CONFIDENCE = 0.1	# minimum probability to filter weak detections
prototext_path = "mobileNetSSD/MobileNetSSD_deploy.prototxt.txt" 	# path to Caffe 'deploy' prototxt file
model_path = "mobileNetSSD/MobileNetSSD_deploy.caffemodel"			# path to Caffe pre-trained model
# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
ONLY_SHOW = ["bottle", "pottedplant"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe(prototext_path, model_path)	# load our serialized model from disk

# set up face model
face_cascade = cv2.CascadeClassifier('expression_model/haarcascade_frontalface_alt.xml')
expression_model = model_from_json(open('expression_model/facial_expression_model_structure.json', 'r').read())
expression_model.load_weights('expression_model/facial_expression_model_weights.h5')
expressions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

webcam = Webcam()
webcam.start()
H, W = None, None
canvas = None
previous_point = None
holdingArray = []
happy_counter = 0

while True:
	# get current frame and image dimensions
	frame = webcam.get_current_frame()
	# frame = cv2.resize(frame, (640, 360))
	if H is None:
		(H, W) = frame.shape[:2]
		canvas = 255 * np.ones(shape=[H, W, 3], dtype=np.uint8)


	# run object detector
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
	net.setInput(blob)
	detections = net.forward()


	# run expression detector
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	# for (x,y,w,h) in faces:
			# if w > 100: #trick: ignore small faces
	if len(faces) > 0:
		(x,y,w,h) = sorted(faces, key=lambda x: x[2], reverse=True)[0]
		cv2.rectangle(frame,(x,y),(x+w,y+h),(64,64,64),2) #highlight detected face
		detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		face_pixels = image.img_to_array(detected_face)
		face_pixels = np.expand_dims(face_pixels, axis = 0)
		face_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		expression_preds = expression_model.predict(face_pixels) #store probabilities of 7 expressions

		# write out emotions + find max emotion
		overlay = frame.copy()	#background of expression list
		opacity = 0.4
		cv2.rectangle(frame,(x+w+10,y-25),(x+w+150,y+115),(64,64,64),cv2.FILLED)
		cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
		cv2.line(frame,(int((x+x+w)/2),y+15),(x+w,y-20),(255,255,255),1)
		cv2.line(frame,(x+w,y-20),(x+w+10,y-20),(255,255,255),1)
		expressionArr = []
		for i in range(len(expression_preds[0])):
			expressionArr.append((expressions[i], round(expression_preds[0][i]*100, 2)))
			expression_text = "%s %s%s" % (expressions[i], round(expression_preds[0][i]*100, 2), '%')
			color = (255,255,255)
			cv2.putText(frame, expression_text, (int(x+w+15), int(y-12+i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
		expression = max(expressionArr, key=lambda item:item[1])[0]
		print(expression)
		if len(holdingArray) < 10:
			holdingArray.append(expression)
		else:
			holdingArray.pop(0)
			holdingArray.append(expression)
			maxEmotion = mostFrequent(holdingArray)
			if maxEmotion in ["fear", "sad", "surprise"]:
				canvas = 255 * np.ones(shape=[H, W, 3], dtype=np.uint8)
			elif maxEmotion == "happy":
				cv2.putText(canvas, "Raghav - Oct '19", (int(W*0.85), (int(H*0.97))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
				cv2.imwrite("files/canvas.jpg", canvas)
				playsound('files/camera-click.wav')
				holdingArray = []

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > MIN_CONFIDENCE: 		# filter out weak detections
			idx = int(detections[0, 0, i, 1]) 	# extract the index of the class label from the detections

			if CLASSES[idx] not in ONLY_SHOW:
				continue

			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H]) # get bounding box
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
				cv2.line(canvas,(canvas_x, canvas_y), previous_point,(0,0,255),7)
			previous_point = canvas_x, canvas_y

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
