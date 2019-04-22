import numpy as np
import cv2
 
 
# load the image
image = cv2.imread("filmerole3_001.jpg")


boundaries = [
	([17, 15, 100], [50, 56, 200]),    # bounds for red
	([40, 20, 20], [225, 60, 60]),     # bounds for blue 
	([150, 150, 150], [255, 255, 255]) # bounds for white
 ]


	
# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
 
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
 
	mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
	mask_rgb[:,:,0] = mask
	mask_rgb[:,:,1] = mask
	mask_rgb[:,:,2] = mask
	

	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)
	print('hello')
