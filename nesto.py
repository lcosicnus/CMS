import cv2
import dlib
import time

carCascade = cv2.CascadeClassifier('myhaar.xml')

tracker = dlib.correlation_tracker()

trackingCars = 0

OUTPUT_SIZE_WIDTH = 1280  
OUTPUT_SIZE_HEIGHT = 720

capture = cv2.VideoCapture('video//prvi.mkv')

cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

cv2.moveWindow("base-image", 0, 100)
cv2.moveWindow("result-image", 720, 100)

cv2.startWindowThread()

rectangleColor = (0, 0, 255)

while True:
	start_time = time.time()
	rc, fullSizeBaseImage = capture.read()
	
	if type(fullSizeBaseImage) == type(None):
		break
	
	baseImage = cv2.resize(fullSizeBaseImage, (760, 520))
		
	resultImage = baseImage.copy()

	if not trackingCars:
		gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
		cars = carCascade.detectMultiScale(gray, 1.1, 5)

		maxArea = 0
		x = 0
		y = 0
		w = 0
		h = 0

		for (xD, yD, wD, hD) in cars:
			if wD * hD > maxArea:
				x = xD
				y = yD
				w = wD
				h = hD
				maxArea = w * h
				
			if maxArea > 0:
				if (x > 135 and x < 295) or (x >= 190 and x < 505) or (x >= 260 and x < 719):
					tracker.start_track(baseImage, dlib.rectangle(x, y, x + w, y + h))
					trackingCars = 1
	
	if trackingCars:
		trackingQuality = tracker.update(baseImage)
		
		if trackingQuality >= 8.75:
			trackedPosition = tracker.get_position()
			
			tx = int (trackedPosition.left())
			ty = int (trackedPosition.top())
			tw = int (trackedPosition.width())
			th = int (trackedPosition.height())
			cv2.rectangle(resultImage, (tx, ty), (tx + tw, ty + th), rectangleColor, 2)
		else:
			trackingCars = 0

			
	fps = 1.0/(time.time() - start_time + 0.000000001)
	#showing video with detected object
	cv2.putText(baseImage, "FPS: " + str(int(fps)), (620, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
	cv2.putText(resultImage, "FPS: " + str(int(fps)), (620, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
	cv2.imshow("result-image", resultImage)
	
	
	if cv2.waitKey(33) == 27:
		break
		
cv2.destroyAllWindows()
capture.release()



























