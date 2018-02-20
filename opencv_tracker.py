import cv2

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = cv2.VideoCapture('video//prvi.mkv')

#WIDTH = 854
#HEIGHT = 480

#def distance(x1, y1, x2, y2):
	
	

def tracker():
	rectangleColor = (0, 0, 255)
	frameCounter = 0
	currentCarID = 0
	
	carTracker = {}
	
	while True:
		rc, image = video.read()
		if type(image) == type(None):
			break
		
#		image = cv2.resize(image, (WIDTH, HEIGHT))
		image = image[150:720, 0:1000]
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
			
			if t_x + t_w >= 800:
				carIDtoDelete.append(carID)
			if t_y <= 0:
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
					#tracker = cv2.Tracker_create('BOOSTING') #tracker
					#tracker = cv2.TrackerBoosting_create() #tracker
					tracker = cv2.TrackerMedianFlow_create() #tracker
					tracker.init(image, bbox)
					carTracker[currentCarID] = tracker	
					currentCarID = currentCarID + 1
					
		for carID in carTracker.keys():
			trackedPosition = carTracker[carID].update(image)
			t_x, t_y, t_w, t_h = trackedPosition[1]
			t_x = int(t_x)
			t_y = int(t_y)
			t_w = int(t_w)
			t_h = int(t_h)
			
			cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 2)
		cv2.imshow('image', resultImage)
				
		if cv2.waitKey(33) == 27:
			break
	
	cv2.destroyAllWindows()

if __name__ == '__main__':
	tracker()
