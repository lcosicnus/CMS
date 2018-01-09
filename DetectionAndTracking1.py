import cv2
import dlib
import time

# initialize a car detector
carCascade = cv2.CascadeClassifier('cars.xml')

# create the tracker
tracker = dlib.correlation_tracker()

trackingCars = 0

# open video 
capture = cv2.VideoCapture('video//prvi.mkv')

# create opencv named window
cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

rectangleColor = (0, 0, 255)

while True:
	start_time = time.time()
	# retrieve image from video
	rc, fullSizeBaseImage = capture.read()
	
	# if there is no more frame terminate
	if type(fullSizeBaseImage) == type(None):
		break
	
	# resize frame to 720x560
	baseImage = cv2.resize(fullSizeBaseImage, (720, 560))
	
	# result image is the image where we mark detected cars
	resultImage = baseImage.copy()

	# if we not tracking a car, then try to find one
	if not trackingCars:
		# gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
		# now use the haar cascade detector to find all faces in the image
		cars = carCascade.detectMultiScale(baseImage, 1.1, 5, 0, (24, 24))
		
		maxArea = 0
		x = 0
		y = 0
		w = 0
		h = 0

		# find largest detected object
		for (xD, yD, wD, hD) in cars:
			if wD * hD > maxArea:
				x = int(xD)
				y = int(yD)
				w = int(wD)
				h = int(hD)
				maxArea = w * h
				
			# if one or more cars are found, initialize the tracker on
			# largest car in the image
			if maxArea > 0:
				if x > 140 and x < 500 and y > 100:
					# initialize the tracker
					tracker.start_track(baseImage, dlib.rectangle(x, y, x + w, y + h))
					# indicator variable, 1 - tracker is active
					trackingCars = 1
	
	# check if the tracker is actively tracking a region in the image
	if trackingCars:
		# update tracker and request information about the quality
		# of the tracking update
		trackingQuality = tracker.update(baseImage)
		
		# if tracking quality is good enough, determine the updated
		# position of the tracked region and draw the rectangle
		if trackingQuality >= 8.75:
			trackedPosition = tracker.get_position()
			
			tx = int (trackedPosition.left())
			ty = int (trackedPosition.top())
			tw = int (trackedPosition.width())
			th = int (trackedPosition.height())
			cv2.rectangle(resultImage, (tx, ty), (tx + tw, ty + th), rectangleColor, 2)
		else:
			# if the quality of the tracking update is not sufficient
			# (e.g. the tracked region moved out of screen, ...) we stop
			# tracking of the object and in the next loop we will find 
			# the largest object in the image
			trackingCars = 0

	# calculating frame rate and put it on result image
	fps = 1.0/(time.time() - start_time + 0.000000001)
	cv2.putText(resultImage, "FPS: " + str(int(fps)), (620, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
	
	cv2.imshow("result-image", resultImage)
	
	# press "esc" to terminate
	if cv2.waitKey(33) == 27:
		break
		
cv2.destroyAllWindows()
capture.release()
