import cv2
import math as m
import numpy as np
import time
import matplotlib.pyplot as plt
import statistics as st

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = cv2.VideoCapture('video//cetvrti.mkv')
lk_params = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def find_distance(r1, c1, r2, c2):
	d = m.sqrt(m.pow(r2 - r1, 2) + m.pow(c2 - c1, 2))
	return d

def find_center(corners):
	x, y = 0, 0
	for corner in corners:
		y += corner[0][1]
		x += corner[0][0]
			
	center_row = int(1.0 * y / len(corners))
	center_col = int(1.0 * x / len(corners))
	
	return center_row, center_col

#------------------------------------------------------------------------------------------
# Speed estimation
#------------------------------------------------------------------------------------------
def estimate_speed(bbox_p, bbox_c, seconds):
	#~ if some of coordinates is 0, estimated speed is 0
	for i in range(0, len(bbox_p)):
		if bbox_p[i] == 0 or bbox_c[i] == 0:
			return 0.000000000000001
	
	#~ calculate center point of previous bounding box
	#~ print x and y coordinates of top left point of bounding box
	#~ print width and height of bounding box and calculated center point of bounding box
	center_p_x = bbox_p[0] + 0.5 * bbox_p[2]
	center_p_y = bbox_p[1] + 0.5 * bbox_p[3]
	
	print('previous: ')
	print('x = ' + str(bbox_p[0]))
	print('y = ' + str(bbox_p[1]))
	print('w = ' + str(bbox_p[2]))
	print('h = ' + str(bbox_p[3]))
	print(center_p_x)
	print(center_p_y)

	#~ calculate center point of current bounding box
	#~ print x and y coordinates of top left point of bounding box
	#~ print width and height of bounding box and calculated center point of bounding box
	center_c_x = bbox_c[0] + 0.5 * bbox_c[2]
	center_c_y = bbox_c[1] + 0.5 * bbox_c[3]

	print('current: ')
	print('x = ' + str(bbox_c[0]))
	print('y = ' + str(bbox_c[1]))
	print('w = ' + str(bbox_c[2]))
	print('h = ' + str(bbox_c[3]))
	print(center_c_x)
	print(center_c_y)
	
	#~ calculate difference between center points
	#~ ppm - pixels per meter, width of car divided by 2
	#~ distance in meters is equal distance in pixels diveded by ppm
	distance_pixels = m.sqrt(m.pow(float(center_c_x - center_p_x), 2) + m.pow(float(center_c_y - center_p_y), 2))
	ppm = ((bbox_c[2] + bbox_p[2]) / 2.0) / 2.0
	distance_meters = distance_pixels / ppm
	
	#~ speed v = s[m]/t[s]
	speed = distance_meters / seconds
	
	#~ print ppm, time, s[pixels], s[meters], velocity
	#~ return calculated speed
	print('ppm = ' + str(ppm))
	print('t = ' + str(seconds) + ' s')
	print('s = ' + str(distance_pixels) + ' pixel')
	print('s = ' + str(distance_meters) + ' m')
	print('v = ' + str("%.2f" % round(speed, 2)) + ' m/s\n')
	return speed

#---------------------------------------------------------------------
# Detection and tracking
#---------------------------------------------------------------------
def tracker():
	red = (0, 0, 255)
	blue = (255, 0, 0)
	green = (0, 255, 0)
	frameCounter = 0
	currentCarID = 0
	
	#~ dictionary for locations
	previous_location = {}
	current_location = {}
	
	#~ dictionary for trackers
	carTracker = {}
	#~ corners = np.array([])
	#~ old_frame_gray = np.ndarray([])

	while True:
		#~ read frame and check it, if it is not frame break
		rc, image = video.read()
		if type(image) == type(None):
			break
		#~ start time of iteration
		#~ crop frame
		#~ copy cropped frame
		#~ add 1 to frame counter
		start = time.time()
		image = image[150:720, 0:950]
		resultImage = image.copy()
		frameCounter = frameCounter + 1
		
		#~ create empty list for trackers you need to delete
		carIDtoDelete = []
		
		#~ iterate trough dictionary of created trackers and if object leave frame delete tracker
		for carID in carTracker.keys():
			#~ get position of bounding box
			trackedPosition = carTracker[carID].update(image)
			t_x, t_y, t_w, t_h = trackedPosition[1]
			t_x = int(t_x)
			t_y = int(t_y)
			t_w = int(t_w)
			t_h = int(t_h)
			
			#~ x_center = t_x + 0.5 * t_w
			#~ y_center = t_y + 0.5 * t_h
			
			#~ if object leave frame add tracker to delete list
			if t_x + t_w >= 750:
				carIDtoDelete.append(carID)
			elif t_y >= 570 or t_y <= 0:
				carIDtoDelete.append(carID) 
			elif t_x <= 0:
				carIDtoDelete.append(carID)
		
		#~ delete all trackers in delete list		
		for carID in carIDtoDelete:
			print('Tracker deleted: ' + str(carID) + '.')
			carTracker.pop(carID, None)
		
		#~ try to detect new object in frame in every 10 frames
		if not (frameCounter % 10):
			#~ convert frame to grayscale
			#~ try to detect new object in frame 
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			cars = carCascade.detectMultiScale(gray, 1.1, 13)
			#~ if object is detected, save it location and calculate center point of bounding box
			for (_x, _y, _w, _h) in cars:
				x = int(_x) + 7
				y = int(_y) + 7
				w = int(_w) - 5
				h = int(_h) - 5
				
				x_bar = x + 0.5 * w
				y_bar = y + 0.5 * h
				
				matchCarID = None
				
				#~ iterate trough tracked objects
				for carID in carTracker.keys():
					#~ get object location and calculate center point of tracked object
					trackedPosition = carTracker[carID].update(image)
					t_x, t_y, t_w, t_h = trackedPosition[1]
					t_x = int(t_x)
					t_y = int(t_y)
					t_w = int(t_w)
					t_h = int(t_h)

					t_x_bar = t_x + 0.5 * t_w
					t_y_bar = t_y + 0.5 * t_h
					
					#~ if condition is true, detected object already have tracker 
					if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
						matchCarID = carID
				
				#~ if detected object don't have tracker yet, create new tracker and add it to tracker dictionary
				#~ save object location for speed estimation
				if matchCarID is None:
					bbox = (x, y, w, h)
					if bbox[0] < 500 or bbox[1] < 100:
						tracker = cv2.TrackerMedianFlow_create()
						tracker.init(image, bbox)
						carTracker[currentCarID] = tracker
						previous_location[currentCarID] = bbox
						currentCarID = currentCarID + 1
						
						#--------------------------------
						# Corner detection
						#--------------------------------
						ROI = gray[y:y + h, x:x + h]
						corners = cv2.goodFeaturesToTrack(ROI, 50, 0.01, 5)
						if type(corners) != type(None):
							corners = np.float32(corners)
							corners[:, 0, 0] += x
							corners[:, 0, 1] += y
							for i in corners:
								x,y = i.ravel()
								if bbox[0] < x < bbox[0] + bbox[2] and bbox[1] < y < bbox[0] + bbox[3]:
									cv2.circle(resultImage, (x, y), 5, 255)
		
		#~ in every frame iterate trough trackers
		for carID in carTracker.keys():
			#~ get position of object
			trackedPosition = carTracker[carID].update(image)
			t_x, t_y, t_w, t_h = trackedPosition[1]
			t_x = int(t_x)
			t_y = int(t_y)
			t_w = int(t_w)
			t_h = int(t_h)
			
			t_x_bar = t_x + 0.5 * t_w
			t_y_bar = t_y + 0.5 * t_h
			bbox = (t_x, t_y, t_w, t_h)
			
			#---------------------------------
			# Optical flow 
			#---------------------------------
			old_corners = corners.copy()
			if len(old_corners) > 0:
				ret, frame = video.read()
				if type(frame) == type(None):
					break
				frame = frame[150:720, 0:950]
				gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				new_corners, st, err = cv2.calcOpticalFlowPyrLK(gray, gray_frame, old_corners, None, **lk_params)
				center_row, center_col = find_center(new_corners)
				cv2.circle(resultImage, (center_col, center_row), 5, blue, 3)
				corners_update = new_corners.copy()
				to_delete = []
				for i in range(len(new_corners)):
					if find_distance(new_corners[i][0][1], new_corners[i][0][0], center_row, center_col) > 30:
						to_delete.append(i)
				corners_update = np.delete(corners_update, to_delete, 0)
				for corner in corners_update:
					if t_x < corner[0][0] < t_x + t_w and t_y < corner[0][1] < t_y + t_w:
						cv2.circle(resultImage, (corner[0][0], corner[0][1]), 5, green, 3)
				
				old_corners = new_corners.copy()
				old_frame_gray = gray_frame.copy()
							
			#~ save location for speed estimation
			#~ draw new rectangle in frame 
			current_location[carID] = bbox
			cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), red, 2)
		
		#~ end time of iteration
		#~ calculate time in seconds
		#~ calculate frame per seconds (fps)
		#~ put fps on frame
		end = time.time()
		seconds = end - start
		fps = 1.0 / seconds
		cv2.putText(resultImage, 'FPS: ' + 	str(int(fps)), (800, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
		
		#~ iterate trough locations
		#~ for i in previous_location.keys():
			#~ save location to local variables
			#~ current location is new previous location
			#~ if coordinates of location is different estimate speed
			#~ bbox_p = previous_location[i]
			#~ bbox_c = current_location[i]
			#~ previous_location[i] = current_location[i]
			#~ if bbox_p != bbox_c:
				#~ speed = estimate_speed(bbox_p, bbox_c, seconds)
		#~ show results
		#~ wait for esc to terminate
		cv2.imshow('image', resultImage)
		if cv2.waitKey(33) == 27:
			break
	#~ close all open
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
