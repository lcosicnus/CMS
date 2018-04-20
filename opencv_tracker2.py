import math as m
import statistics as st
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = cv2.VideoCapture('video//prvi.mkv')
lk_params = dict(winSize = (15, 15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
velocity = {}


def find_distance(x1, y1, x2, y2):
	d = m.sqrt(m.pow(float(x2) - x1, 2) + m.pow(float(y2) - y1, 2))
	return d

def distance(w1, w2):
		realWidth = 1.8
		f = 0.0022
		imageWidth = 800
		objectWidth = (w1 + w2) / 2.0
		sensorWidth = 0.005376
		distanceToObject = (f * realWidth * imageWidth) / (objectWidth * sensorWidth)
		return distanceToObject
		# print 'Distance to object: ' + str(distanceToObject) + ' m.'

def find_center(corners):
	x, y = 0, 0
	for corner in corners:
		y += corner[0][1]
		x += corner[0][0]
	center_row = int(1.0 * y / len(corners))
	center_col = int(1.0 * x / len(corners))
	return (center_col, center_row)

def timeToCollision(v, d):
    
	ttc = (d / v)
	return ttc

# Speed estimation
def estimate_speed(c1, c2, seconds, w1, w2, carID):
	n = 0
	distance_pixels = find_distance(c1[0], c1[1], c2[0], c2[1])
	mpp = 1.8 / ((w1 + w2) / 2.0) 
	meters = distance_pixels * mpp
	v = meters / (seconds)
	v_kph = v * 3.6

	if carID in velocity.keys():
		i = len(velocity[carID])
		speed = []
		if i < 5:
			velocity[carID].append(v_kph)
		else:
			speed = velocity[carID][-5:]
			avg_speed = np.mean(speed)
			# print avg_speed
			if abs(v_kph - avg_speed) < 2:
				velocity[carID].append(v_kph)
				# distance(w1, w2)
				print 'Car ' + str(carID) + ' new speed is: ' + str(velocity[carID][-1:])
	else:
    		
			velocity[carID] = [v_kph]
	return v
	
# Detection and tracking
def tracker():
	red = (0, 0, 255)
	blue = (255, 0, 0)
	green = (0, 255, 0)
	frameCounter = 0
	currentCarID = 0
	
	# dictionary for locations
	previous_location = {}
	current_location = {}
	
	# dictionary for trackers
	carTracker = {}

	# dictionary for corners
	corners1 = {}
	corners2 = {}
	corners_update = {}
	corners_center = {}
	old_corners_center = {}
	width1 = {}
	width2 = {}
	speed = {}
	time1 = {}
	time2 = {}
	while True:
		# read frame and check it, if it is not frame break

		rc, image = video.read()
		if type(image) == type(None):
			break
		# start time of iteration
		# crop frame
		# copy cropped frame
		# add 1 to frame counter
		# start = time.time()
		image = image[150:600, 150:950]
		resultImage = image.copy()
		frameCounter = frameCounter + 1
		
		# create empty list for trackers you need to delete
		carIDtoDelete = []
		
		# iterate trough dictionary of created trackers and if object leave frame delete tracker
		for carID in carTracker.keys():
			# get position of bounding box
			trackedPosition = carTracker[carID].update(image)
			t_x, t_y, t_w, t_h = trackedPosition[1]
			t_x = int(t_x)
			t_y = int(t_y)
			t_w = int(t_w)
			t_h = int(t_h)
			
			# x_center = t_x + 0.5 * t_w
			# y_center = t_y + 0.5 * t_h
			
			# if object leave frame add tracker to delete list
			if t_x + t_w >= 750:
				carIDtoDelete.append(carID)
			elif t_y >= 570 or t_y <= 0:
				carIDtoDelete.append(carID) 
			elif t_x <= 0:
				carIDtoDelete.append(carID)
		
		# delete all trackers in delete list
		for carID in carIDtoDelete:
			# print 'Tracker deleted: ' + str(carID) + '.'
			# print 'Current location deleted: ' + str(carID) + '.'
			# print 'Previous location deleted: ' + str(carID) + '.'
			# print 'Corners 1 deleted: ' + str(carID) + '.'
			# print 'Corners 2 deleted: ' + str(carID) + '.'
			# print 'Width 1 deleted: ' + str(carID) + '.'
			# print 'Width 2 deleted: ' + str(carID) + '.'
			# print '\n'
			carTracker.pop(carID, None)
			current_location.pop(carID, None)
			previous_location.pop(carID, None)
			corners1.pop(carID, None)
			corners2.pop(carID, None)
			width1.pop(carID, None)
			width2.pop(carID, None)
			time1.pop(carID, None)
			time2.pop(carID, None)
			corners_center.pop(carID, None)
			old_corners_center.pop(carID, None)
		
		# try to detect new object in frame in every 10 frames
		if not (frameCounter % 10):
			# convert frame to grayscale
			# try to detect new object in frame 
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			cars = carCascade.detectMultiScale(gray, 1.15, 15)
			# if object is detected, save it location and calculate center point of bounding box
			for (_x, _y, _w, _h) in cars:
				x = int(_x) + 7
				y = int(_y) + 7
				w = int(_w) - 5
				h = int(_h) - 5
				
				x_bar = x + 0.5 * w
				y_bar = y + 0.5 * h
				
				matchCarID = None
				
				# iterate trough tracked objects
				for carID in carTracker.keys():
					# get object location and calculate center point of tracked object
					trackedPosition = carTracker[carID].update(image)
					t_x, t_y, t_w, t_h = trackedPosition[1]
					t_x = int(t_x)
					t_y = int(t_y)
					t_w = int(t_w)
					t_h = int(t_h)

					t_x_bar = t_x + 0.5 * t_w
					t_y_bar = t_y + 0.5 * t_h
					
					# if condition is true, detected object already have tracker 
					if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
						matchCarID = carID
				
				# if detected object don't have tracker yet, create new tracker and add it to tracker dictionary
				# save object location for speed estimation
				if matchCarID is None:
					bbox = (x, y, w, h)
					if bbox[0] + bbox[2] < 400 and bbox[1] < 100 and bbox[0] > 70:
						tracker = cv2.TrackerMedianFlow_create()
						tracker.init(image, bbox)
						carTracker[currentCarID] = tracker
						previous_location[currentCarID] = bbox
						ROI = gray[y:y + h, x:x + h]
						corners1[currentCarID] = cv2.goodFeaturesToTrack(ROI, 10, 0.25, 5)
						corners1[currentCarID][:, 0, 0] += x
						corners1[currentCarID][:, 0, 1] += y
						for i in corners1[currentCarID]:
							x, y = i.ravel()
							cv2.circle(resultImage, (x, y), 5, green, thickness = -1)
						#--
						width1[currentCarID] = bbox[2]
						old_corners_center[currentCarID] = find_center(corners1[currentCarID])
						#--
						time1[currentCarID] = time.time()
						currentCarID = currentCarID + 1
		
		# in every frame iterate trough trackers
		for carID in carTracker.keys():
			# get position of object
			trackedPosition = carTracker[carID].update(image)
			t_x, t_y, t_w, t_h = trackedPosition[1]
			t_x = int(t_x)
			t_y = int(t_y)
			t_w = int(t_w)
			t_h = int(t_h)
			
			t_x_bar = t_x + 0.5 * t_w
			t_y_bar = t_y + 0.5 * t_h
			bbox = (t_x, t_y, t_w, t_h)
			width2[carID] = bbox[2]

			if len(corners1[carID]):
				ret, frame = video.read()
				if type(frame) == type(None):
					break
				frame = frame[150:600, 150:950]
				gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				corners2[carID], st, err = cv2.calcOpticalFlowPyrLK(gray, gray2, corners1[carID], None, **lk_params)
				corners_center[carID] = find_center(corners2[carID])
				# print(corners_center[carID])
				cv2.circle(resultImage, corners_center[carID], 5, blue, thickness = -1)
				for corner in corners2[carID]:
					cv2.circle(resultImage, (corner[0][0], corner[0][1]), 5, green, -1)
				corners1[carID] = corners2[carID].copy()
				time2[carID] = time.time()
				gray = gray2.copy()
									
			# save location for speed estimation
			# draw new rectangle in frame 
			current_location[carID] = bbox
			cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), red, 2)
		
		#--
		# iterate trough locations
		for i in corners_center.keys():
			sec = time2[i] - time1[i]
			if old_corners_center[i] != corners_center[i] and sec >= 0.01 and sec < 0.1:
				v = estimate_speed(old_corners_center[i], corners_center[i], sec, width1[i], width2[i], i)
				print v
				d = distance(width1[i], width2[i])
				cv2.putText(resultImage, 'Distance: ' + str(int(d)) + ' m.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
				cv2.putText(resultImage, 'TTC: ' + str(int(timeToCollision(v, d))) + ' s.', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
			old_corners_center[i] = corners_center[i]
			time1[i] = time2[i]
			# if len(width1):
			width1[i] = width2[i]
		#--
		# show results
		# wait for esc to terminate
		if cv2.waitKey(33) == 27:
    			break
		cv2.imshow('image', resultImage)
		
	# close all open
	cv2.destroyAllWindows()
if __name__ == '__main__':
	tracker()
