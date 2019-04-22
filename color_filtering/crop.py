import cv2
import os

filenames = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
filenames.sort()

for filename in filenames:

	if '.jpg' in filename and not '(1)' in filename:
	
		img = cv2.imread(filename)
		cropped = img[85:1030,:,:]
		cv2.imwrite('cropped/' + filename[:14] + '_cropped.jpg', cropped)
