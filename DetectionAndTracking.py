import cv2
import numpy as np

def diffUpDown(img):
	height, widht, depth = img.shape
	half = height / 2
	top = img[0:half, 0:widht]
	bottom = img[half:half + half, 0:widht]
	top = cv2.flip(top, 1)
	bottom = cv2.resize(bottom, (32, 64))
	top = cv2.resize(top, (32, 64))
	return (mse(top, bottom))

def diffLeftRight(img):
	height, widht, depth = img.shape
	half = widht / 2
	left = img[0:height, 0:half]
	right = img[0:height, half:half + half - 1]
	right = cv2.flip(right, 1)
	left = cv2.resize(left, (32, 64))
	right = cv2.resize(right, (32, 64))
	return (mse(left, right))
	
def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err
	
def isNewRoi(rx, ry, rw, rh, rectangles):
	for r in rectangles:
		if abs(r[0] - rx) < 40 and abs(r[1] - ry) < 40:
			return False
	return True
	
def detectRegionsOfInteres(frame, cascade):
	scaleDown = 2
	frameHeight, frameWidth, frameDepth = frame.shape
	
	frame = cv2.resize(frame, (frameWidth / scaleDown, frameHeight / scaleDown))
	frameHeight, frameWidth, frameDepth = frame.shape
	
	# cars = cascade.detectMultiScale(frame, 1.2, 1)
	cars = cascade.detectMultiScale(frame, 1.2, 10, 0, (24, 24))
	
	newRegions = []
	minY = 24
	
	for (x, y, w, h) in cars:
		roi = [x, y, w, h]
		roiImage = frame[y:y + h, x:x + w]
		
		carWidht = roiImage.shape[0]
		if y > minY:
			diffX = diffLeftRight(roiImage)
			diffY = round(diffUpDown(roiImage))
			# print 'diffx = ' + str(diffX)
			# print 'diffy = ' + str(diffY)
			
			if diffX > 3000 and diffX < 6000 and diffY > 15000:
				rx, ry, rw, rh = roi
				newRegions.append([rx * scaleDown, ry * scaleDown, rw * scaleDown, rh * scaleDown])
	
	return newRegions

def detectCars(filename):
	rectangles = []
	cascade = cv2.CascadeClassifier('myhaar.xml')
	vc = cv2.VideoCapture(filename)
	if vc.isOpened():
		rval, frame = vc.read()
	else:
		rval = False
	
	roi = [0, 0, 0, 0]
	frameCount = 0
	
	while rval:
		rval, frame = vc.read()
		frameHeight, frameWidth, frameDepth = frame.shape
		
		newRegions = detectRegionsOfInteres(frame, cascade)
		for region in newRegions:
			if isNewRoi(region[0], region[1], region[2], region[3], rectangles):
				rectangles.append(region)
				
		for r in rectangles:
			cv2.rectangle(frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 0, 255), 2)
			
		frameCount = frameCount + 1
		if frameCount > 5:
			frameCount = 0
			rectangles = []
			
		cv2.imshow('Result', frame)
		if cv2.waitKey(33) == 27:
			vc.release()
			break

detectCars('video//prvi.mkv')


	
	
	