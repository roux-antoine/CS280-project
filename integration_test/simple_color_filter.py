import numpy as np
import cv2
import os

## CONSTANTS AND PARAMETERS ##

color_bounds =  {'blue':  [[ 40,  20,  20], [255,  60,  60]],
                 'white': [[150, 150, 150], [255, 255, 255]]}

colors =  {'blue':  [ 255,  0,   0],
           'white': [255, 255, 255]}

blur_kernel_size = 31
blur_std_dev = 0
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 0
params.maxThreshold = 255
params.thresholdStep = 10
params.filterByArea = True
params.minArea = 130
params.minDistBetweenBlobs = 80
params.filterByInertia = False
params.filterByConvexity = False
params.filterByCircularity = False


#################################

if __name__ == '__main__':


	## Generating the filenames ##
	os.chdir('test_images')
	filenames = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
	filenames.sort()

	for filename in filenames:
		## Open image and blur it ##
	
		image = cv2.imread(filename)
		blurred = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), blur_std_dev)
		# cv2.imshow("images", blurred)
		# cv2.waitKey(0)

		for color in color_bounds.keys():

			## Isolate color patches ##
		
			lower = np.array(color_bounds[color][0])
			upper = np.array(color_bounds[color][1])
			mask = cv2.inRange(blurred, lower, upper)
			# output = cv2.bitwise_and(blurred, blurred, mask = mask) # useless??
			inverted_mask = cv2.bitwise_not(mask)

			## Detect color blobs ##

			detector = cv2.SimpleBlobDetector_create(params)
			keypoints = detector.detect(inverted_mask)
	 	
			## Draw the keypoints ##
			for keypoint in keypoints:
				cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size), colors[color], 10)
			

			# Show keypoints
		#cv2.imshow("img", image)
		#cv2.waitKey(0)

		cv2.imwrite('processed_' + filename, image)
	

