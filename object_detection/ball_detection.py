import cv2
import numpy as np
import os
# Read image
# raw = cv2.imread("test_images/image3.jpg", cv2.IMREAD_GRAYSCALE)
# im = cv2.GaussianBlur(raw, (9,9), 0)

# cv2.imshow('test',im)
# cv2.waitKey(0)

# _, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()





params.minDistBetweenBlobs = 0


# cv2.imshow('test',im)
# cv2.waitKey(0)

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255
params.thresholdStep = 10
# Filter by Area.
params.filterByArea = True
params.minArea = 130

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
    
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.5

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)

os.chdir('filmrole3_frames')
filenames = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
filenames.sort()

for filename in filenames:
	# Detect blobs.
	raw = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	im = cv2.GaussianBlur(raw, (3,3), 0)
	# cv2.imshow("test", im)
	_, im = cv2.threshold(im, 125, 255, cv2.THRESH_BINARY_INV)


	keypoints = detector.detect(im)
	print(keypoints)
	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
	# the size of the circle corresponds to the size of blob

	im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	# Show blobs
	cv2.imshow("Keypoints", im_with_keypoints)
	cv2.waitKey(0)