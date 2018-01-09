import cv2
import dlib
import time
import threading

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = cv2.VideoCapture('video//prvi.mkv')

WIDTH = 720
HEIGHT = 560

# def carNumber(carNum, cID):
	# time.sleep(2)
	# carNum[cID] = 'Car ' + str(cID)
	
def trackMultipleObjects():
	rectangleColor = (0, 0, 255)
	frameCounter = 0
	currentCarID = 0
	fps = 0
	
	carTracker = {}
	carNumbers = {}
	
	while True:
		start_time = time.time()
		rc, image = video.read()
		if type(image) == type(None):
			break
		
		image = cv2.resize(image, (WIDTH, HEIGHT))
		resultImage = image.copy()
		
		frameCounter = frameCounter + 1
		
		carIDtoDelete = []
		for carID in carTracker.keys():
			trackingQuality = carTracker[carID].update(image)
			
			if trackingQuality < 7:
				carIDtoDelete.append(carID)
				
		for carID in carIDtoDelete:
			print 'Removing carID ' + str(carID) + ' from list of trackers.'
			carTracker.pop(carID, None)
		
		if not (frameCounter % 10):
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			cars = carCascade.detectMultiScale(gray, 1.1, 12, 0, (24, 24))
			
			for (_x, _y, _w, _h) in cars:
				x = int(_x)
				y = int(_y)
				w = int(_w)
				h = int(_h)
			
				x_bar = x + 0.5 * w
				y_bar = y + 0.5 * h
				
				matchCarID = None
			
				for carID in carTracker.keys():
					trackedPosition = carTracker[carID].get_position()
					
					t_x = int(trackedPosition.left())
					t_y = int(trackedPosition.top())
					t_w = int(trackedPosition.width())
					t_h = int(trackedPosition.height())
					
					t_x_bar = t_x + 0.5 * t_w
					t_y_bar = t_y + 0.5 * t_h
				
					if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
						matchCarID = carID
				
				if matchCarID is None:
					print 'Creating new tracker ' + str(currentCarID)
					
					tracker = dlib.correlation_tracker()
					tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
					
					carTracker[currentCarID] = tracker
					
					# t = threading.Thread(target = carNum, args = (carNumbers, currentCarID))
					# t.start()
					
					currentCarID = currentCarID + 1
					
		for carID in carTracker.keys():
			trackedPosition = carTracker[carID].get_position()
					
			t_x = int(trackedPosition.left())
			t_y = int(trackedPosition.top())
			t_w = int(trackedPosition.width())
			t_h = int(trackedPosition.height())
			
			cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 2)
			
			# if carID in carNumbers.keys():
				# cv2.putText(resultImage, carNumbers[carID], (int(t_x + t_w/2), int(t_y)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			# else:
				# cv2.putText(resultImage, 'Detecting...', (int(t_x + t_w/2), int(t_y)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		
		end_time = time.time()
		if not (end_time == start_time):
			fps = 1.0/(end_time - start_time)
		cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
		cv2.imshow('result', resultImage)
				
		if cv2.waitKey(33) == 27:
			break
	
	cv2.destroyAllWindows()

if __name__ == '__main__':
	trackMultipleObjects()