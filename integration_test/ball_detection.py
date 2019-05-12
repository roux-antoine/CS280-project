import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import numpy as np
# Read image
# raw = cv2.imread("test_images/image3.jpg", cv2.IMREAD_GRAYSCALE)
# im = cv2.GaussianBlur(raw, (9,9), 0)

# cv2.imshow('test',im)
# cv2.waitKey(0)

# _, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
# Setup SimpleBlobDetector parameters.

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def arrowplot(axes, x, y, narrs=50, dspace=0.5, direc='neg', \
                          hl=0.5, hw=10, c='blue'): 
    ''' narrs  :  Number of arrows that will be drawn along the curve

        dspace :  Shift the position of the arrows along the curve.
                  Should be between 0. and 1.

        direc  :  can be 'pos' or 'neg' to select direction of the arrows

        hl     :  length of the arrow head 

        hw     :  width of the arrow head        

        c      :  color of the edge and face of the arrow head  
    '''
    # r is the distance spanned between pairs of points
    r = [0]
    for i in range(1,len(x)):
        dx = x[i]-x[i-1] 
        dy = y[i]-y[i-1] 
        r.append(np.sqrt(dx*dx+dy*dy))
    r = np.array(r)

    # rtot is a cumulative sum of r, it's used to save time
    rtot = []
    for i in range(len(r)):
        rtot.append(r[0:i].sum())
    rtot.append(r.sum())

    # based on narrs set the arrow spacing
    aspace = r.sum() / narrs

    if direc is 'neg':
        dspace = -1.*abs(dspace) 
    else:
        dspace = abs(dspace)

    arrowData = [] # will hold tuples of x,y,theta for each arrow
    arrowPos = aspace*(dspace) # current point on walk along data
                                 # could set arrowPos to 0 if you want
                                 # an arrow at the beginning of the curve

    ndrawn = 0
    rcount = 1 
    while arrowPos < r.sum() and ndrawn < narrs:
        x1,x2 = x[rcount-1],x[rcount]
        y1,y2 = y[rcount-1],y[rcount]
        da = arrowPos-rtot[rcount]
        theta = np.arctan2((x2-x1),(y2-y1))
        ax = np.sin(theta)*da+x1
        ay = np.cos(theta)*da+y1
        arrowData.append((ax,ay,theta))
        ndrawn += 1
        arrowPos+=aspace
        while arrowPos > rtot[rcount+1]: 
            rcount+=1
            if arrowPos > rtot[-1]:
                break

    # could be done in above block if you want
    for ax,ay,theta in arrowData:
        # use aspace as a guide for size and length of things
        # scaling factors were chosen by experimenting a bit

        dx0 = np.sin(theta)*hl/2. + ax
        dy0 = np.cos(theta)*hl/2. + ay
        dx1 = -1.*np.sin(theta)*hl/2. + ax
        dy1 = -1.*np.cos(theta)*hl/2. + ay

        if direc is 'neg' :
          ax0 = dx0 
          ay0 = dy0
          ax1 = dx1
          ay1 = dy1 
        else:
          ax0 = dx1 
          ay0 = dy1
          ax1 = dx0
          ay1 = dy0 

        axes.annotate('', xy=(ax0, ay0), xycoords='data',
                xytext=(ax1, ay1), textcoords='data',
                arrowprops=dict( headwidth=hw, frac=1., ec=c, fc=c))

    axes.plot(x,y, color = c)
    axes.set_xlim(x.min()*.9,x.max()*1.1)
    axes.set_ylim(y.min()*.9,y.max()*1.1)


params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255
params.thresholdStep = 10
# Filter by Area.
params.filterByArea = True
params.minArea = 80

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.85

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.85
    
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.4

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)

os.chdir('filmrole3_frames')
filenames = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
filenames.sort()

prev = []
result = []
for filename in filenames:
	# Detect blobs.
	raw = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	crop = raw[92:1030, 0:1919]
	# cv2.imshow("Keypoints", crop)
	# cv2.waitKey(0)
	im = cv2.GaussianBlur(crop, (3,3), 0)
	# cv2.imshow("test", im)
	_, im = cv2.threshold(im, 175, 255, cv2.THRESH_BINARY_INV)

	p = []
	keypoints = detector.detect(im)

	# if len(keypoints)!=0:
		# im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		# cv2.imshow("Keypoints", im_with_keypoints)
		# cv2.waitKey(0)
	if len(keypoints)==1:
		p = keypoints[0].pt
		prev = p
	elif len(keypoints)==2:
		d1 = distance(keypoints[0].pt, prev)
		d2 = distance(keypoints[1].pt, prev)
		if(d1 > d2):
			p = keypoints[0].pt
			prev = p
		else:
			p = keypoints[1].pt
			prev = p
	if p != []:
		result.append(p)
		print(p)

	# if(len(result) > 1):
	# 	x,y = zip(*result)
	# 	x = list(x)
	# 	y = list(y)
	# 	print(x)
	# 	print(y)
	# 	# plt.scatter(x, y)
	# 	# plt.plot(x, y, '-o')
	# 	fig = plt.figure()
	# 	axes = fig.add_subplot(111)
	# 	arrowplot(axes, np.asarray(x), np.asarray(y), narrs=len(result)) 
	# 	plt.show()

x,y = zip(*result)
x = list(x)
y = list(y)
print(x)
print(y)
# plt.scatter(x, y)
# plt.plot(x, y, '-o')
fig = plt.figure()
axes = fig.add_subplot(111)
arrowplot(axes, np.asarray(x), np.asarray(y), narrs=len(result)) 
plt.show()

	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
	# the size of the circle corresponds to the size of blob

	# im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	# # Show blobs
	# cv2.imshow("Keypoints", im_with_keypoints)
	# cv2.waitKey(0)