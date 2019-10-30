# python real_time_object_detection.py
import numpy as np
import time
import cv2
from webcam import Webcam
from keras.preprocessing import image
from keras.models import model_from_json, load_model
from playsound import playsound
from sklearn.metrics import pairwise
import argparse
# global variables
MIN_CONFIDENCE = 0.2	# minimum probability to filter weak detections
prototext_path = "mobileNetSSD/MobileNetSSD_deploy.prototxt.txt" 	# path to Caffe 'deploy' prototxt file
model_path = "mobileNetSSD/MobileNetSSD_deploy.caffemodel"			# path to Caffe pre-trained model
# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
ONLY_SHOW = ["bottle", "pottedplant"]
PALETTE = {	1: {"name": "red", "value": (0,0,255)},
			2: {"name": "blue", "value": (255,0,0)},
			3: {"name": "green", "value": (0,255,0)},
			4: {"name": "yellow", "value": (0,255,255)},
			5: {"name": "magenta", "value": (255,0,255)}	}
current_color = 1
bg = None

# in case video needs to be saved
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--artist", default="Raghav",
	help="name of the person creating the painting")
args = vars(ap.parse_args())


# To find the running average over the background
def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)    # compute weighted average, accumulate it and update the background

# To segment the region of hand in the image
def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)     # find the absolute difference between background and current frame
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]     # threshold the diff image so that we get the foreground
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     # get the contours in the thresholded image
    if len(cnts) == 0:
        return None
    else:
        segmented = max(cnts, key=cv2.contourArea) # based on contour area, get the maximum contour which is the hand
        return (thresholded, segmented)

# To count the number of fingers in the segmented hand region
def count(thresholded, segmented):
    chull = cv2.convexHull(segmented)     # find the convex hull of the segmented hand region
    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])
    # find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]
    radius = int(0.8 * maximum_distance)     # calculate the radius of the circle with 80% of the max euclidean distance obtained
    circumference = (2 * np.pi * radius)     # find the circumference of the circle
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")     # take out the circular region of interest which has palm and the fingers
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)     # draw the circular ROI
    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    (_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)     # compute the contours in the circular ROI

    count = 0
    for c in cnts:     # loop through the contours found
        (x, y, w, h) = cv2.boundingRect(c) # compute the bounding box of the contour
        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed 5% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1
    return count

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


# set up
face_cascade = cv2.CascadeClassifier('expression_model/haarcascade_frontalface_alt.xml')
expression_model = model_from_json(open('expression_model/facial_expression_model_structure.json', 'r').read())
expression_model.load_weights('expression_model/facial_expression_model_weights.h5')
expressions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe(prototext_path, model_path)	# load our serialized model from disk

webcam = Webcam()
webcam.start()
time.sleep(2)

H, W = None, None
canvas = None
previous_point = None
holdingArray = []
happy_counter = 0

accumWeight = 0.5     # initialize accumulated weight
top, right, bottom, left = 10, 350, 225, 590     # region of interest (ROI) coordinates
num_frames = 0     # region of interest (ROI) coordinates
calibrated = False     # calibration indicator

while True:
	# get current frame and image dimensions
	frame = webcam.get_current_frame()
	frame = cv2.flip(frame, 1)
	output = frame.copy() # draw everying on this
	# frame = cv2.resize(frame, (640, 360))
	if H is None:
		(H, W) = frame.shape[:2]
		canvas = 255 * np.ones(shape=[H, W, 3], dtype=np.uint8)
		signature = args["artist"] + " - Oct 19"
		cv2.putText(canvas, signature, (int(W*0.85), (int(H*0.97))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
		top, right, bottom, left = 0, W - 251, 250, W - 1


	# gesture recognition stuff
	roi = frame[top:bottom, right:left]         # get the ROI
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)
	if num_frames < 30:
		run_avg(gray, accumWeight)
		if num_frames == 1:
			print("[STATUS] please wait! calibrating...")
		elif num_frames == 29:
			print("[STATUS] calibration successfull...")
		num_frames += 1
		continue
	else:
		hand = segment(gray)  # segment the hand region
		if hand is not None:
			(thresholded, segmented) = hand
			cv2.drawContours(output, [segmented + (right, top)], -1, (0, 0, 255))
			fingers = count(thresholded, segmented) # number of fingers detected
			if fingers > 0 and fingers < 6:
				current_color = fingers
			color_text = "You have chosen color " + str(current_color) + " - " + PALETTE[current_color]["name"]
			cv2.putText(output, color_text, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, PALETTE[current_color]["value"], 2)
	cv2.rectangle(output, (left, top), (right, bottom), (0,255,0), 2)


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
		cv2.rectangle(output,(x,y),(x+w,y+h),(64,64,64),2) #highlight detected face
		detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		face_pixels = image.img_to_array(detected_face)
		face_pixels = np.expand_dims(face_pixels, axis = 0)
		face_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		expression_preds = expression_model.predict(face_pixels) #store probabilities of 7 expressions

		# write out emotions + find max emotion
		overlay = output.copy()	#background of expression list
		opacity = 0.4
		cv2.rectangle(output,(x+w+10,y-25),(x+w+150,y+115),(64,64,64),cv2.FILLED)
		cv2.addWeighted(overlay, opacity, output, 1 - opacity, 0, frame)
		cv2.line(output,(int((x+x+w)/2),y+15),(x+w,y-20),(255,255,255),1)
		cv2.line(output,(x+w,y-20),(x+w+10,y-20),(255,255,255),1)
		expressionArr = []
		for i in range(len(expression_preds[0])):
			expressionArr.append((expressions[i], round(expression_preds[0][i]*100, 2)))
			expression_text = "%s %s%s" % (expressions[i], round(expression_preds[0][i]*100, 2), '%')
			color = (255,255,255)
			cv2.putText(output, expression_text, (int(x+w+15), int(y-12+i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
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
				signature = args["artist"] + " - Oct 19"
				cv2.putText(canvas, signature, (int(W*0.85), (int(H*0.97))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
			elif maxEmotion == "happy":
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
				cv2.line(canvas,(canvas_x, canvas_y), previous_point, PALETTE[current_color]["value"],7)
			previous_point = canvas_x, canvas_y

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(output, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(output, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

			# for making a collage of labels
			# cv2.putText(canvas, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			# cv2.rectangle(canvas, (startX, startY), (endX, endY), COLORS[idx], 2)

	cv2.imshow("output", output)
	cv2.imshow("canvas", canvas)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

# cleanup
webcam.end()
