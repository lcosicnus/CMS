import cv2
import numpy as np

video = 'video//prvi.mkv'
cascade_src = 'myhaar.xml'
width = 854
height = 480
red = (0, 0, 200)

cap = cv2.VideoCapture(video)
cascade = cv2.CascadeClassifier(cascade_src)
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

def resize(frame):
	resized_frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
	return resized_frame

def rgb2gray(frame):
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	return gray_frame

def detect(frame):
	all_roi = []
	gray_frame = rgb2gray(frame)
	cars = cascade.detectMultiScale(gray_frame, 1.1, 13, 0, (24,24))
	for x, y, w, h in cars:
		all_roi.append((x, y, x + w, y + h))
		
	cv2.imshow('video', frame)
	cv2.waitKey(1)
	return all_roi

def track(all_roi, all_roi_hist):
	for k in range(0, 30):
		ret, frame = cap.read()
		if type(frame) == type(None):
			return -1
		i = 0
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		
		for roi_hist in all_roi_hist:
			back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
			(r, all_roi[i]) = cv2.CamShift(back_proj, all_roi[i], termination)
			
			for j in range(0, 4):
				if all_roi[i][j] < 0:
					return -1
			pts = np.int32(cv2.cv.BoxPoints(r))
			cv2.polylines(frame, [pts], True, red, 2)
			i += 1
		
		cv2.imshow('video', frame)
		cv2.waitKey(1)
	return 1
		

def calc_hist(frame, all_roi):
	all_roi_hist = []
	for pts in all_roi:
		roi = frame[pts[1]:pts[-1], pts[0]:pts[2]]
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
		roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
		all_roi_hist.append(roiHist)
	return all_roi_hist

def show():
	for k in range(0, 2):
		ret, frame = cap.read()
		if type(frame) == type(None):
			break
		cv2.imshow('video', frame)
		if cv2.waitKey(1) == 27:
			break
def main():
	i = 0
	while True:
		if i % 2 == 0:
			all_roi = []
			all_hsv_roi = []
			ret, frame = cap.read()
			if type(frame) == type(None):
				cap.release()
				cv2.destroyAllWindows()
				return
			
			frame = resize(frame)
			all_roi = detect(frame)
				
			if len(all_roi) != 0:
				all_roi_hist = calc_hist(frame, all_roi)
				i += 1
			else:
				show()
		else:
			error = track(all_roi, all_roi_hist)
			if error == -1:
				cap.release()
				cv2.destroyAllWindows()
				return
			i += 1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
if __name__ == '__main__':
	main()
