import cv2
import numpy as np
import time
import math as m

cap = cv2.VideoCapture('video//prvi.mkv')
cascade = cv2.CascadeClassifier('myhaar.xml')

cv2.namedWindow('Detecting', cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow('Tracking', cv2.WINDOW_AUTOSIZE)

cv2.moveWindow('Detecting', 100, 100)
# cv2.moveWindow('Tracking', 500, 100)

cv2.startWindowThread()

frame_counter = 0
# corners = []
green = (0, 200, 0)
blue = (200, 0, 0)
red = (0, 0 , 200)
LK_parameters = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def find_distance(r1, c1, r2, c2):
	d = m.sqrt(m.pow(r2 - r1, 2) + m.pow(c2 - c1, 2))
	return d

def detect(frame):
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	detected = cascade.detectMultiScale(gray_frame, 1.1, 13, 0, (24, 24))
	
	return detected
	
def find_center(corners):
	x, y = 0, 0
	for corner in corners:
			y += corner[0][1]
			x += corner[0][0]
			
	center_row = int(1.0 * y / len(corners))
	center_col = int(1.0 * x / len(corners))
	
	return center_row, center_col

def detect_features(x, y, w, h):
	ret, frame = cap.read()
	
	if type(frame) == type(None):
		return None
	
	corners = []
	
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	ROI = gray_frame[x:x + w, y:y + h]
	corners = cv2.goodFeaturesToTrack(ROI, 50, 0.01, 5)
	
	if type(corners) == type(None):
		return None
	
	corners[:, 0, 0] += x
	corners[:, 0, 1] += y

	return corners
	
def track_features(old_frame, old_corners):
	old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	found = 0
	while True:
		ret, frame = cap.read()
		if type(frame) == type(None):
			break
		
		frame = cv2.resize(frame, (640, 360))
		img = frame.copy()
		if len(old_corners) > 0 and found == 0:
			new_frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
			corners, st, err = cv2.calcOpticalFlowPyrLK(old_frame_gray, new_frame_gray, old_corners, None, **LK_parameters)
			
			center_row, center_col = find_center(corners)
			
			cv2.circle(frame, (center_col, center_row), 5, blue, 5)

			corners_update = corners.copy()
			
			toDelete = []
			
			for i in range(len(corners)):
				if find_distance(corners[i][0][1], corners[i][0][0], center_row, center_col) > 90:
					toDelete.append(i)
			
			corners_update = np.delete(corners_update, toDelete, 0)
			
			for corner in corners_update:
				cv2.circle(frame, (int(corner[0][0]), int(corner[0][1])), 5, green)
			if len(corners_update) < 5:
				cars = detect(frame)
				for x, y, w, h in cars:
					found += 1
					if found == 1:
						corners_update = detect_features(x + 4, y + 4, w -7, h - 7)
						if type(corners_update) == type(None):
							break
						center_row, center_col = find_center(corners_update)
						break
					# break
				# track_features(frame, new_corners)
			if type(corners_update) != type(None):
				if len(corners_update) > 0:
					for corner in corners_update:
						if corner[0][0] > 640 or corner[0][0] < 0 or corner[0][1] > 360 or corner[0][1] < 0:
							return
			
				if len(corners_update) > 0:
					ctr, rad = cv2.minEnclosingCircle(corners_update)
					cv2.circle(frame, (int(ctr[0]), int(ctr[1])), int(rad), red, 5)
			else:
				return
			
			old_frame_gray = new_frame_gray.copy()
			old_corners = corners_update.copy()
		
		cv2.imshow('Detecting', frame)
		
		found = 0
		if cv2.waitKey(33) == 27:
			break
		
def main():
	frame_counter = 0
	found = 0
	corners = np.array([])
	while True:
		ret, frame = cap.read()
		if type(frame) == type(None):
			break
		frame = cv2.resize(frame, (640, 360))
		if not (frame_counter % 10):
			cars = detect(frame)
			
			for x, y, w, h in cars:
				found += 1
				if found == 1:
					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
					corners = detect_features(x, y, w, h)
					if type(corners) == type(None):
						break
					for corner in corners:
						cv2.circle(frame, (int(corner[0][0]), int(corner[0][1])), 3, green, thickness = 2, lineType = 4)
				else:
					break
		if found:
			track_features(frame, corners)
		else:
			cv2.imshow('Detecting', frame)
		frame_counter += 1
		found = 0
		
		if cv2.waitKey(33) == 27:
			break
	
if __name__ == '__main__':
	main()
	cv2.destroyAllWindows()
	cap.release()
