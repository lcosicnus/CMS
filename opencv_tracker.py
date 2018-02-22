import cv2
import math as m
import numpy as np
import time
import matplotlib.pyplot as plt
import statistics as st

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = cv2.VideoCapture('video//prvi.mkv')


def estimate_speed(bbox_p, bbox_c, seconds):
	#~ print('p: ' + str(bbox_p))
	#~ print('c: ' + str(bbox_c))	
	
	center_p_x = bbox_p[0] + 0.5 * bbox_p[2]
	center_p_y = bbox_p[1] + 0.5 * bbox_p[3]
	
	print('previous: ')
	print('x = ' + str(bbox_p[0]))
	print('y = ' + str(bbox_p[1]))
	print('w = ' + str(bbox_p[2]))
	print('h = ' + str(bbox_p[3]))
	print(center_p_x)
	print(center_p_y)

	center_c_x = bbox_c[0] + 0.5 * bbox_c[2]
	center_c_y = bbox_c[1] + 0.5 * bbox_c[3]

	print('current: ')
	print('x = ' + str(bbox_c[0]))
	print('y = ' + str(bbox_c[1]))
	print('w = ' + str(bbox_c[2]))
	print('h = ' + str(bbox_c[3]))
	print(center_c_x)
	print(center_c_y)

	distance_pixels = m.sqrt(m.pow(float(center_c_x - center_p_x), 2) + m.pow(float(center_c_y - center_p_y), 2))
	ppm = ((bbox_c[2] + bbox_p[2]) / 2.0) / 2.0
	distance_meters = distance_pixels / ppm
	
	speed = distance_meters / seconds
	
	print('ppm = ' + str(ppm))
	print('t = ' + str(seconds) + ' s')
	print('s = ' + str(distance_pixels) + ' pixel')
	print('s = ' + str(distance_meters) + ' m')
	print('v = ' + str("%.2f" % round(speed, 2)) + ' m/s\n')
	return speed

def tracker():
	rectangleColor = (0, 0, 255)
	frameCounter = 0
	currentCarID = 0
	
	previous_location = {}
	current_location = {}
	
	carTracker = {}
	v = []
	median = []
	
	while True:
		rc, image = video.read()
		if type(image) == type(None):
			break
		
		start = time.time()
		image = image[150:720, 0:950]
		resultImage = image.copy()
		
		frameCounter = frameCounter + 1
		
		carIDtoDelete = []
		
		for carID in carTracker.keys():
			trackedPosition = carTracker[carID].update(image)
			#print(trackedPosition)
			t_x, t_y, t_w, t_h = trackedPosition[1]
			t_x = int(t_x)
			t_y = int(t_y)
			t_w = int(t_w)
			t_h = int(t_h)
			
			x_center = t_x + 0.5 * t_w
			y_center = t_y + 0.5 * t_h
			
			if t_x + t_w >= 750:
				carIDtoDelete.append(carID)
				
		for carID in carIDtoDelete:
			print('Tracker deleted: ' + str(carID) + '.')
			carTracker.pop(carID, None)
		
		if not (frameCounter % 10):
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			cars = carCascade.detectMultiScale(gray, 1.1, 13)
			for (_x, _y, _w, _h) in cars:
				x = int(_x) + 5
				y = int(_y) + 5
				w = int(_w) - 5
				h = int(_h) - 5
				
				x_bar = x + 0.5 * w
				y_bar = y + 0.5 * h
				
				
				matchCarID = None
			
				for carID in carTracker.keys():
					trackedPosition = carTracker[carID].update(image)
					t_x, t_y, t_w, t_h = trackedPosition[1]
					t_x = int(t_x)
					t_y = int(t_y)
					t_w = int(t_w)
					t_h = int(t_h)

					t_x_bar = t_x + 0.5 * t_w
					t_y_bar = t_y + 0.5 * t_h
										
					if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
						matchCarID = carID
								
				if matchCarID is None:
					bbox = (x, y, w, h)
					if bbox[0] < 500 or bbox[1] < 100:
						#tracker = cv2.Tracker_create('BOOSTING') #tracker
						#tracker = cv2.TrackerBoosting_create() #tracker
						tracker = cv2.TrackerMedianFlow_create() #tracker
						tracker.init(image, bbox)
						carTracker[currentCarID] = tracker	
						previous_location[currentCarID] = bbox
						currentCarID = currentCarID + 1
					
		for carID in carTracker.keys():
			trackedPosition = carTracker[carID].update(image)
			t_x, t_y, t_w, t_h = trackedPosition[1]
			t_x = int(t_x)
			t_y = int(t_y)
			t_w = int(t_w)
			t_h = int(t_h)
			
			t_x_bar = t_x + 0.5 * t_w
			t_y_bar = t_y + 0.5 * t_h
			bbox = (t_x, t_y, t_w, t_h)
			corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
			if type(corners) != type(None):
				corners = np.int0(corners)
			for i in corners:
				x,y = i.ravel()
				if t_x < x < t_x + t_w and t_y < y < t_y + t_h:
					cv2.circle(resultImage, (x, y), 5, 255, -1)
			
			current_location[carID] = bbox
				
			cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 2)
		
		
		end = time.time()
		seconds = end - start
		fps = 1.0 / seconds
		cv2.putText(resultImage, 'FPS: ' + 	str(int(fps)), (800, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
		
		for i in previous_location.keys():
			if frameCounter % 1 == 0:
				bbox_p = previous_location[i]
				bbox_c = current_location[i]
				
				previous_location[i] = current_location[i]
				if bbox_p != bbox_c:
					estimate_speed(bbox_p, bbox_c, seconds)
		
		cv2.imshow('image', resultImage)
		if cv2.waitKey(33) == 27:
			break
			
	cv2.destroyAllWindows()
if __name__ == '__main__':
	tracker()

#-------------------------------------------------------------------------------------
#Corner detection
#-------------------------------------------------------------------------------------
#~ gray_ROI = gray[y:y + h, x:x + w]
#~ corners = cv2.goodFeaturesToTrack(gray_ROI, 25, 0.01, 10)
#~ if type(corners) != type(None):
	#~ corners = np.int0(corners)
#~ for i in corners:
	#~ x,y = i.ravel()
	#~ cv2.circle(resultImage, (x, y), 3, 255, 2)

#----------------------------
# Statistics
#----------------------------
#~ med = []
#~ average = []
#~ i = range(0, len(v))
#~ avg = np.mean(v)
#~ m = st.median(v)
#~ for x in i:
	#~ average.append(avg)
	#~ med.append(m)
#~ plt.plot(i, v)
#~ plt.plot(i, average)
#~ plt.plot(i, med)
#~ plt.legend(['speed', 'average', 'median'])
#~ plt.show()
