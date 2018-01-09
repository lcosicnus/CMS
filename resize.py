from PIL import Image
import os, sys
import cv2

path = r'C:\Users\Luka\Desktop\Slike\cars_train'
savePath = r'C:\Users\Luka\Desktop\dasar_haartrain\positive\rawdata'

dirs = os.listdir(path)


counter = 907
width = 512

for item in dirs:
	name = item.split('.')
	if os.path.isfile(path + '\\' + item):
		im = Image.open(path + '\\' + item)
		file, ext = os.path.splitext(path + '\\' + item)
		
		widthPercent = (width / float(im.size[0]))
		height = int(float(im.size[1])*float(widthPercent))
		imResize = im.resize((width, height), Image.ANTIALIAS)
		
		if counter < 1000:
			name[0] = '00' + str(counter)
		else:
			name[0] = '0' + str(counter)
			
		imResize.save(savePath + '\\' + name[0] + '.bmp', 'BMP', quality = 100)
		counter = counter + 1