import numpy as np
import cv2
 
 
# load the image
image = cv2.imread("filmerole3_001_cropped.png")


boundaries = [
	([40, 20, 20], [255, 60, 60]),     # bounds for blue 
	([150, 150, 150], [255, 255, 255]) # bounds for white
 ]

# low pass filter it

blurred = cv2.GaussianBlur(image, (31,31), 0)
cv2.imshow("images", blurred)
cv2.waitKey(0)

	
# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
 
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(blurred, lower, upper)
	output = cv2.bitwise_and(blurred, blurred, mask = mask)
 

	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	params.minThreshold = 0
	params.maxThreshold = 255
	params.thresholdStep = 10


	# Filter by Area.
	params.filterByArea = True
	params.minArea = 130

	# Filter by Circularity
	params.filterByCircularity = False
	# params.minCircularity = 0.1

	# Filter by Convexity
	params.filterByConvexity = False
	#params.minConvexity = 0.87

	# Filter by Inertia
	params.filterByInertia = False
	#params.minInertiaRatio = 0.01

	params.minDistBetweenBlobs = 80

	inverted_mask = cv2.bitwise_not(mask)




	detector = cv2.SimpleBlobDetector_create(params)
	 
	# Detect blobs.
	keypoints = detector.detect(inverted_mask)
	 
	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	im_with_keypoints = cv2.drawKeypoints(blurred, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	 
	# Show keypoints
	cv2.imshow("Keypoints", im_with_keypoints)
	cv2.waitKey(0)
	

	# show the images
	#cv2.imshow("images", np.hstack([blurred, output]))
	#cv2.waitKey(0)
	#print('hello')
