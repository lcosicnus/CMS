import cv2
import numpy as np
import time

#path to classifier
#path to video file
cascade_src = 'myhaar.xml'
video_src = 'video//drugi.mkv'

#read video
#load classifier from file
cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

fps = 0

#defining borders of ROI
x1 = 0
y1 = 160
v = 720
hc = v - y1

#reading video frame by frame
while True:
	#starting time of loop iteration
	start_time = time.time()
	
	#read frame
	ret, img = cap.read()
	if (type(img) == type(None)):
		break
		
	#crop frame to get ROI
	img = img[y1:y1+hc, x1:x1+720]

	#convert to gray scale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	#Detect objects with different dimensions in frame
	#output of function is list of rectangles
	#parameters: image, scaleFactior, minNeighbors, flags, minSize, maxSize
	cars = car_cascade.detectMultiScale(gray, 1.1, 13, 0, (24, 24))
	
	#drawing rectangle around detected object
	#defining regions where object will show up
	for (x, y, w, h) in cars:
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
	
	#calculation of frame rate
	fps = 1.0/(time.time() - start_time)
	
	#showing video with detected object
	cv2.putText(img, "FPS: " + str(int(fps)), (620, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
	cv2.imshow('video', img)
	
	#press "esc" key to terminate
	if cv2.waitKey(33) == 27:
		break;

cv2.destroyAllWindows()
cap.release()
