import cv2
import dlib
import time

rectangleColor = (0, 0, 255)
frameCounter = 0
currentCarID = 0

carTrackers = {}
carNames = {}

carCascade = cv2.CascadeClassifier('myhaar.xml')

capture = cv2.VideoCapture('video//prvi.mkv')
def trackMultiple():
	while True:
		rc, fullSizeBaseImage = capture.read()
		
		if type(fullSizeBaseImage) == type(None):
			break
		
		baseImage = cv2.resize(fullSizeBaseImage, (720, 560))
		resultImage = baseImage.copy()
		
		frameCounter += 1
		
		carIDsToDelete = []
		for cID in carTrackers.keys():
			trackingQuality = carTrackers[cID].update(baseImage)
			
			if trackingQuality < 7:
				carIDsToDelete.append(cID)
			
		for cID in carIDsToDelete:
			print 'Remove car ID: ' + str(cID) + ' from list of trackers.'
			carTrackers.pop(cID, None)
		
		if (frameCounter % 10) == 0:
			gray = cv2.svtColor(baseImage, cv2.COLOR_BGR2GRAY)
			cars = carCascade.detectMultiScale(gray, 1.3, 5)
			
			for (xD, yD, wD, hD) in cars:
				x = int(xD)
				y = int(yD)
				w = int(wD)
				h = int(hD)
				
				xCenter = x + 0.5 * w
				yCenter = y + 0.5 * h
				
				matchedCarID = None
				
				for cID in carTrackers.keys():
					trackedPosition = carTrackers[cID].get_position()
					
					xT = int(trackedPosition.left())
					yT = int(trackedPosition.top())
					wT = int(trackedPosition.width())
					hT = int(trackedPosition.height())
					
					trackedXCenter = xT + 0.5 * wT
					trackedYCenter = yT + 0.5 * hT
					
					if ((xT <= xCenter <= (xT + wT)) and 
						(yT <= yCenter <= (yT + hT)) and
						(x <= trackedXCenter <= (x + w)) and
						(y <= trackedYCenter <= (y + h))):
						matchedCarID = cID
					
				if matchedCarID is None:
					print 'Creating new tracker: ' + str(currentCarID) + '.'
					
					tracker = dlib.correlation_tracker()
					tracker.start_track(baseImage, dlib.rectangle(x, y, x + w, y + h)
					carTrackers[currentCarID] = tracker
					currentCarID += 1
					
		for cID in carTrackers.keys():
			tPosition = carTrackers[cID].get_position()
			
			trackedX = int(tPosition.left())
			trackedY = int(tPosition.top())
			trackedW = int(tPosition.width())
			trackedH = int(tPosition.height())
				
			cv2.rectangle(resultImage, (trackedX, trackedY), (trackedX + trackedW, trackedY + trackedH), rectangleColor, 2)
					
		cv2.imshow('Result: ', resultImage)
		
		if cv2.waitKey(33) == 27:
			break
if __name__ == '__main__':
	trackMultiple()