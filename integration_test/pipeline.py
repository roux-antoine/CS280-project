# Object Detection Demo
#migrated from the python notebook version
# change eve

#
## IMPORTS
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib
import seaborn as sns
import planar
import pandas as pd
import time

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
# print(matplotlib.get_backend())
from PIL import Image
import cv2



## CONSTANTS AND PARAMETERS ##
# for color filter parameters

color_bounds =  {'blue':  [[ 40,  20,  20], [255,  60,  60]],
                 'white': [[150, 150, 150], [255, 255, 255]]}

colors =  {'blue':  [ 255,  0,   0],
           'white': [255, 255, 255]}

blur_kernel_size = 7
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
font = cv2.FONT_HERSHEY_SIMPLEX



# This is needed since the utils folder is stored in
# /Users/chestermu/Documents/Tensorflow/models/research/object_detection/utils

###########################################################################
##CHANGE DIRECTORY HERE
# sys.path.append("/Users/chestermu/Documents/Tensorflow/models/research/")
sys.path.append("/home/likai/tensorflow/models/research")
###########################################################################
from object_detection.utils import ops as utils_ops


if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


## Object detection imports
# This is needed since the utils folder is stored in
# /Users/chestermu/Documents/Tensorflow/models/research/object_detection/utils
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util



## Model preparation

# What model to download.
MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'


MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# modified: change directory
###########################################################################
##CHANGE DIRECTORY HERE
# PATH_TO_LABELS = os.path.join('../../../../../Documents/Tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('/home/likai/tensorflow/models/research/object_detection/data/', 'mscoco_label_map.pbtxt')
###########################################################################


###########################################################################
## UNCOMMENT HERE TO DOWNLOAD MODEL
## Download model
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())
###########################################################################


## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


## Loading label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

## Helper Code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


###########################################################################
##CHANGE DIRECTORY HERE
PATH_TO_TEST_IMAGES_DIR = 'filmrole3_frames'
###########################################################################

TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'filmerole3_00{}.jpg'.format(i)) for i in range(1, 10) ] + [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'filmerole3_0{}.jpg'.format(i)) for i in range(10, 100) ] + [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'filmerole3_{}.jpg'.format(i)) for i in range(100, 602) ]
# TEST_IMAGE_PATHS =[ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'filmerole3_0{}.jpg'.format(i)) for i in range(95, 98) ]
#print(TEST_IMAGE_PATHS)
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

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


def plot_player_pos(x_coord, y_coord, color, image_number):

    height = 892  #the height of the stadium image
    width = 1346  #the width of the stadium image

    y_coord = height - y_coord # to account for different convention on origin of field

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

    my_ax.scatter(x_coord, y_coord, color=color)

    plt.axis('equal')
    plt.xlim(0, width)
    plt.ylim(0, height)
    if color == 'red':
        color = 'white'
    plt.savefig('player_pos/' + color + '/player_pos_' + color + '_' + str(image_number) + '.png')
    my_ax.clear()

def plot_heatmap(x_coord, y_coord, color, image_number):

    height = 892  #the height of the stadium image
    width = 1346  #the width of the stadium image

    y_coord = height - y_coord # to account for different convention on origin of field

    # plt.scatter(x_coord, y_coord)
    # plt.savefig('heatmap_' + color + '.png')
    # plt.clf()

    my_ax = sns.kdeplot(x_coord, y_coord, shade = "True", color = color, n_levels = 30)

    ## begin of image border lines ##
    lineCenter1 = matplotlib.patches.ConnectionPatch([width/4, 0], [width/3, height], "data", "data")
    lineCenter2 = matplotlib.patches.ConnectionPatch([3*width/4, 0], [2*width/3, height], "data", "data")
    lineLeft = matplotlib.patches.ConnectionPatch([width/2, 0], [width/3, height], "data", "data")
    lineRight = matplotlib.patches.ConnectionPatch([width/2, 0], [2*width/3, height], "data", "data")
    # my_ax.plot([width/3,   width/4],   [0, height], '--', c='k')  # line center 1
    # my_ax.plot([2*width/3, 3*width/4], [0, height], '--', c='k')  # line center 2
    # my_ax.plot([width/3,   width/2],   [0, height], '--', c='k')  # line left
    # my_ax.plot([2*width/3, width/2],   [0, height], '--', c='k')  # line right

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

    plt.axis('equal')
    plt.xlim(0, width)
    plt.ylim(0, height)
    if color == 'red':
        color = 'white'
    plt.savefig('heatmap/' + color + '/heatmap_' + color + '_' + str(image_number) + '.png')
    my_ax.clear()

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def white_or_blue(box):
  # cv2.imshow('boximage',box)
  # cv2.waitKey(0)
  blurred = cv2.GaussianBlur(box, (blur_kernel_size, blur_kernel_size), blur_std_dev)
  # cv2.imshow('blurredboximage',blurred)
  # cv2.waitKey(0)

  colorkeypts = []
  for color in color_bounds.keys():
    ## Isolate color patches ##
    lower = np.array(color_bounds[color][0])
    upper = np.array(color_bounds[color][1])
    mask = cv2.inRange(blurred, lower, upper)
    # output = cv2.bitwise_and(blurred, blurred, mask = mask) # useless??
    inverted_mask = cv2.bitwise_not(mask)
    
    # cv2.imshow('inverted_mask',inverted_mask)
    # cv2.waitKey(0)
    ## Detect color blobs ##

    detector = cv2.SimpleBlobDetector_create(params)
    colorkeypts.append(detector.detect(inverted_mask))

    # for keypoint in keypoints:
    #   cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size), colors[color], 10)
  #need to output the label    

  if colorkeypts[0] and not colorkeypts[1]:
    return list(color_bounds.keys())[0]
  elif not colorkeypts[0] and colorkeypts[1]:
    return list(color_bounds.keys())[1]
  elif colorkeypts[0] and colorkeypts[1]:
    a = max(colorkeypts[0], key = lambda pt: pt.size)
    b = max(colorkeypts[1], key = lambda pt: pt.size)
    return list(color_bounds.keys())[0] if a.size>b.size else list(color_bounds.keys())[1]
  else:
    return None


def detect_ball(img):
  params = cv2.SimpleBlobDetector_Params()
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
    
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gaussian = cv2.GaussianBlur(gray, (3,3), 0)
  _, inv = cv2.threshold(gaussian, 125, 255, cv2.THRESH_BINARY_INV)
  
  keypoints = detector.detect(inv)
  # print(keypoints)
  p = []
  if(len(keypoints)!=0):
	  p = keypoints[0].pt
  return p


###################

blue_players_map = []  #list of all the positions of all blue players in all images (expressed in map frame, ie top frame)
white_players_map = [] #list of all the positions of all white players in all images (expressed in map frame, ie top frame)

H = homography_solve()

for image_nbr, image_path in enumerate(TEST_IMAGE_PATHS):

  start_time = time.time()

  image = Image.open(image_path)
  width, height = image.size
  #Size 1920*945
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
  
  boxes_array = output_dict['detection_boxes'].copy()
  # print(boxes_array.shape)

  #convert pil img to cv2 img
  pilimg = image.convert('RGB') 
  cvimg = np.array(pilimg) 
  cvimg = cvimg[:, :, ::-1].copy()

  blue_player_pos_top_frame_this_image = []
  white_player_pos_top_frame_this_image = []

  for box in boxes_array:
    for pos in range(len(box)):
      if pos%2 == 0:
        box[pos] *= height
      else:
        box[pos] *= width

    box = box.astype(int)
    if box[2]-box[0] <= 0 or box[3]-box[1] <= 0:
      continue
    else:
      # print(box)
      boximg = cvimg[box[0]:box[2],box[1]:box[3]]
      detect = white_or_blue(boximg)

      if detect == 'blue':
          player_pos_cam_frame = np.array([(box[1]+box[3])/2, (box[0]+box[2])/2])
          player_pos_top_frame = homography_transform(player_pos_cam_frame, H)
          blue_player_pos_top_frame_this_image.append(player_pos_top_frame)
          blue_players_map.append([int(player_pos_top_frame[0]), int(player_pos_top_frame[1])])
          
      elif detect == 'white':
          player_pos_cam_frame = np.array([(box[1]+box[3])/2, (box[0]+box[2])/2])
          player_pos_top_frame = homography_transform(player_pos_cam_frame, H)
          white_player_pos_top_frame_this_image.append(player_pos_top_frame)
          white_players_map.append([int(player_pos_top_frame[0]), int(player_pos_top_frame[1])])
      else:
          continue

  np.savetxt('csv/blue/blue_pos_' + str(image_nbr) + '.csv', np.asarray(blue_player_pos_top_frame_this_image), delimiter=',')
  np.savetxt('csv/white/white_pos_' + str(image_nbr) + '.csv', np.asarray(white_player_pos_top_frame_this_image), delimiter=',')


  # print(str(len(blue_players_map)))
  # print(str(len(white_players_map)))
  print("Number of Blue Player Detected:" + str(len(blue_player_pos_top_frame_this_image)))
  print("Number of white Player Detected:" + str(len(white_player_pos_top_frame_this_image)))

  if len(blue_players_map) != 0:
    plot_heatmap(np.array(blue_players_map)[:,0], np.array(blue_players_map)[:,1], 'blue', image_nbr)

  if len(blue_players_map) != 0:
    plot_heatmap(np.array(white_players_map)[:,0], np.array(white_players_map)[:,1], 'red', image_nbr)

  if len(blue_player_pos_top_frame_this_image) != 0:
    plot_player_pos(np.array(blue_player_pos_top_frame_this_image)[:,0], np.array(blue_player_pos_top_frame_this_image)[:,1], 'blue', image_nbr)
  
  if len(white_player_pos_top_frame_this_image) != 0:
    plot_player_pos(np.array(white_player_pos_top_frame_this_image)[:,0], np.array(white_player_pos_top_frame_this_image)[:,1], 'red', image_nbr)

  cv2.rectangle(cvimg,(box[1],box[0]),(box[3],box[2]),(0,255,0),3)
  cv2.putText(cvimg,detect,(box[1],box[2]), font, 1,(255,255,255),2,cv2.LINE_AA)

  print("Currently Processing Image Number " + str(image_nbr))

  elapsed = time.time() - start_time
  print("Processing Time" + str(elapsed))
      
  cv2.imshow('image',cvimg)
  cv2.waitKey(0)


#######################################################
  # # Visualization of the results of a detection.
  # vis_util.visualize_boxes_and_labels_on_image_array(
  #     image_np,
  #     output_dict['detection_boxes'],
  #     output_dict['detection_classes'],
  #     output_dict['detection_scores'],
  #     category_index,
  #     instance_masks=output_dict.get('detection_masks'),
  #     use_normalized_coordinates=True,
  #     line_thickness=8)
  # plt.figure(figsize=IMAGE_SIZE)
  # plt.imshow(image_np)
  # plt.show()
  # plt.savefig('im1.jpg')


