import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
import matplotlib


# Read image
# raw = cv2.imread("test_images/image3.jpg", cv2.IMREAD_GRAYSCALE)
# im = cv2.GaussianBlur(raw, (9,9), 0)

# cv2.imshow('test',im)
# cv2.waitKey(0)

# _, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
# Setup SimpleBlobDetector parameters.
def homography_solve():
    """
    u = points in the video
    v = points on the map
    """
    v = np.array([[674, 318],
                  [546, 454],
                  [676, 580],
                  [800, 452]])

    u = np.array([[942, 301],
                  [453, 402],
                  [944, 517],
                  [1426, 402]])

    U = np.zeros((8, 2*u.shape[0]))
    for k in range(2*u.shape[1]):
        U[2*k,:] =   [u[k,0], u[k,1], 1, 0, 0, 0, -v[k, 0]*u[k, 0], -v[k, 0]*u[k, 1]]
        U[2*k+1,:] = [0, 0, 0, u[k,0], u[k,1], 1, -v[k, 1]*u[k, 0], -v[k, 1]*u[k, 1]]

    V = []
    for k in range(2*v.shape[1]):
        V.append(v[k,0])
        V.append(v[k,1])

    V = np.array(V)

    h = np.matmul(np.linalg.inv(np.matmul(U.T, U)), np.matmul(U.T, V))

    H = np.array([[h[0], h[1], h[2]],
                  [h[3], h[4], h[5]],
                  [h[6], h[7], 1]])
    return H

def homography_transform(u, H):
    u = np.append(u, 1)
    u_homogeneous = np.array(u).reshape(3,1)
    v_homogeneous = np.matmul(H, u)
    return [v_homogeneous[0]/v_homogeneous[2], v_homogeneous[1]/v_homogeneous[2]]


def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def arrowplot(axes, x, y, narrs=50, dspace=0.5, direc='neg', \
                          hl=0.1, hw=5, c='blue'): 
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

def plot_ball_pos(x_coord, y_coord):

    height = 892  #the height of the stadium image
    width = 1346  #the width of the stadium image

    y_coord = height- np.array(y_coord)
    # y_coord = height - y_coord # to account for different convention on origin of field

    # my_fig = plt.Figure()
    # my_ax = plt.Axes(fig=my_fig, rect=[0, 0, width, height])
    my_ax = sns.kdeplot([-1, 0, width+1, width], [0, -1, height, height+1], shade = "False", color = 'green', n_levels = 0)


    ## begin of image border lines ##
    lineCenter1 = matplotlib.patches.ConnectionPatch([width/4, 0], [width/3, height], "data", "data")
    lineCenter2 = matplotlib.patches.ConnectionPatch([3*width/4, 0], [2*width/3, height], "data", "data")
    lineLeft = matplotlib.patches.ConnectionPatch([width/2, 0], [width/3, height], "data", "data")
    lineRight = matplotlib.patches.ConnectionPatch([width/2, 0], [2*width/3, height], "data", "data")
    # my_ax.plot([  width/3,   width/4], [0, height], '--', c='k')  # line center 1
    # my_ax.plot([2*width/3, 3*width/4], [0, height], '--', c='k')  # line center 2
    # my_ax.plot([  width/3,   width/2], [0, height], '--', c='k')  # line left
    # my_ax.plot([2*width/3,   width/2], [0, height], '--', c='k')  # line right

    ## Drawing the white lines ##
    pitch = plt.Rectangle([0, 0], width = width, height = height, fill = False)
    leftPenalty = plt.Rectangle([0, 0.28*height], width = 0.12*width, height = 0.45*height, fill = False)
    rightPenalty = plt.Rectangle([0.88*width, 0.28*height], width = 0.12*width, height = 35.3/80*height, fill = False)
    midline = matplotlib.patches.ConnectionPatch([0.5*width, 0], [0.5*width, height], "data", "data")
    leftSixYard = plt.Rectangle([0, 0.4*height], width = 0.04*width, height = 0.2*height, fill = False)
    rightSixYard = plt.Rectangle([0.96*width, 0.4*height], width = 0.04*width, height = 0.2*height, fill = False)
    centreCircle = plt.Circle((0.5*width, 0.5*height), 0.09*width,color="black", fill = False)
    centreSpot = plt.Circle((0.5*width, 0.5*height), 0.005*width,color="black")
    leftPenaltySpot = plt.Circle((0.08*width, 0.5*height), 0.005*width, color="black")
    rightPenaltySpot = plt.Circle((0.92*width, 0.5*height), 0.005*width, color="black")

    elements = [pitch, leftPenalty, rightPenalty, midline, leftSixYard, rightSixYard, centreCircle, centreSpot, rightPenaltySpot, leftPenaltySpot]

    for i in elements:
        my_ax.add_patch(i)

    # my_ax.scatter(x_coord, y_coord, color="blue")
    arrowplot(my_ax, np.asarray(x_coord), np.asarray(y_coord), narrs=len(x_coord)) 

    plt.axis('equal')
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.show()
    # plt.savefig('ball_pos/' + 'ball_pos_' + '.png')
    # my_ax.clear()




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
    # cv2.imshow("raw", raw)
    # cv2.waitKey(0)
    crop = raw[92:1030, 0:1920]
    # cv2.imshow("Cropped", crop)
    # cv2.waitKey(0)
    im = cv2.GaussianBlur(crop, (3,3), 0)
    # cv2.imshow("test", im)
    _, im = cv2.threshold(im, 175, 255, cv2.THRESH_BINARY_INV)

    p = []
    keypoints = detector.detect(im)

    if len(keypoints)!=0:
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(0)
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

H = homography_solve()

result_map = []
for i in range(len(result)):
    x = result[i][0]
    y = result[i][1] + 92
    
    homo_x, homo_y = homography_transform([x, y], H)

    if homo_y >= 0 and homo_y < 892:
        result_map.append([homo_x, homo_y])



x,y = zip(*result_map)
x = list(x)
y = list(y)
print(x)
print(y)
# plt.scatter(x, y)
# plt.plot(x, y, '-o')

# fig = plt.figure()
# axes = fig.add_subplot(111)

plot_ball_pos(x, y)

# arrowplot(my_ax, np.asarray(x), np.asarray(y), narrs=len(result)) 
# plt.show()