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

from timeit import default_timer as timer




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
sys.path.append("/Users/chestermu/Documents/Tensorflow/models/research/")
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
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'


MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# modified: change directory
###########################################################################
##CHANGE DIRECTORY HERE
PATH_TO_LABELS = os.path.join('../../Documents/Tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
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

## Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.

###########################################################################
##CHANGE DIRECTORY HERE
PATH_TO_TEST_IMAGES_DIR = 'test_images'
###########################################################################

TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(2, 3) ]
#print(TEST_IMAGE_PATHS)
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

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


def middle_pt(point):
  num = len(point)
  if num ==0:
    return 0,0
  sum0 = 0
  sum1 = 1
  for pt in point:
    sum0+=pt[0][0]
    sum1+=pt[0][1]
  return sum1//num,sum0//num


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

# optical flow tracking
# for each ten frames:
#   if first frame: run detectors, and get all the box coordinates
#     pass these box coordinates to the optical flow thing to look for features and track


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 6,#100
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (10,10), #15,15#
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


cap = cv2.VideoCapture("./filmrole/filmrole5.avi")
#cap = cv2.VideoCapture("./filmrole/test_2.mov")

color = np.random.randint(0,255,(100,3))

# #initialization

# ret, image = cap.read()
# height, width, _ = image.shape
# image_np_expanded = np.expand_dims(image, axis=0)
# output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
# boxes_array = output_dict['detection_boxes'].copy()
# p0 = cv2.goodFeaturesToTrack(image, mask = None, **feature_params) #useHarrisDetector = True
# box = boxes_array[0]
# for pos in range(len(box)):
#   if pos%2 == 0:
#     box[pos] *= height
#   else:
#     box[pos] *= width
# box = box.astype(int)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# graybox = gray[box[0]-10:box[2]+10,box[1]-10:box[3]+10]
# #graybox = gray[box[0]-40:box[2]+40,box[1]-40:box[3]+40]
# p0 = cv2.goodFeaturesToTrack(graybox, mask = None, **feature_params) #useHarrisDetector = True
# for pt in p0:
#   x = int(pt[0][0]+box[1]-10)
#   y = int(pt[0][1]+box[0]-10)
#   marked = cv2.circle(image,(x,y),5,color[10].tolist(),-1)

# box_height = box[2]-box[0]+20
# box_width = box[3]-box[1]+20

################################
# previous debugging
# cv2.imshow('graybox',marked)
# k = cv2.waitKey(0)
# cv2.destroyAllWindows()
# cap.release()
# print(box)
# print(box_height)
# print(box_width)

box = None
p0 = None
gray = None
mask = None

for i in range(200):
  if i %15 == 0:
    ret, image = cap.read()
    height, width, _ = image.shape
    image_np_expanded = np.expand_dims(image, axis=0)
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    boxes_array = output_dict['detection_boxes'].copy()
    box = boxes_array[0]
    for pos in range(len(box)):
      if pos%2 == 0:
        box[pos] *= height
      else:
        box[pos] *= width
    box = box.astype(int)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    graybox = gray[box[0]-10:box[2]+10,box[1]-10:box[3]+10]
    #graybox = gray[box[0]-40:box[2]+40,box[1]-40:box[3]+40]
    # mask_use = np.zeros(gray.shape)
    # temp = np.ones(gray.shape)
    # temp*=255
    #mask_use[box[0]-10:box[2]+10,box[1]-10:box[3]+10] = temp[box[0]-10:box[2]+10,box[1]-10:box[3]+10]#255
 


    p0 = cv2.goodFeaturesToTrack(graybox, mask = None, **feature_params)
    for pt in p0:
      x = int(pt[0][0]+box[1]-10)
      y = int(pt[0][1]+box[0]-10)
      marked = cv2.circle(image,(x,y),5,color[10].tolist(),-1)
    box_height = box[2]-box[0]+20
    box_width = box[3]-box[1]+20
    mask = np.zeros_like(image)

  else:
    
    ret, image = cap.read()
    height, width, _ = image.shape
    new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    newbox = new_gray[box[0]-10:box[2]+10,box[1]-10:box[3]+10]
    #newbox = gray[box[0]-40:box[2]+40,box[1]-40:box[3]+40]
    # print(graybox.shape)
    # print(newbox.shape)
    average_height,average_width = middle_pt(p0)
    p1, st, err = cv2.calcOpticalFlowPyrLK(graybox, newbox, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (int(a+box[1]-10),int(b+box[0]-10)),(int(c+box[1]-10),int(d+box[0]-10)), color[i].tolist(), 2)
        frame = cv2.circle(image,(int(a+box[1]-10),int(b+box[0]-10)),5,color[i].tolist(),-1)
        cv2.rectangle(frame,(box[1]-10,box[0]-10),(box[3]+10,box[2]+10),(0,255,0),3)
        # mask = cv2.line(mask, (int(a+box[1]-40),int(b+box[0]-40)),(int(c+box[1]-40),int(d+box[0]-40)), color[i].tolist(), 2)
        # frame = cv2.circle(image,(int(a+box[1]-40),int(b+box[0]-40)),5,color[i].tolist(),-1)
        # cv2.rectangle(frame,(box[1]-40,box[0]-40),(box[3]+40,box[2]+40),(0,255,0),3)
    img = cv2.add(frame,mask)
    


    cv2.imshow('frame',img)

    k = cv2.waitKey(50) & 0xff
    if k == 27:
        break

    graybox = newbox.copy()
    p0 = good_new.reshape(-1,1,2)
    new_average_height,new_average_width = middle_pt(p0)
    # print(p0)
    print(average_height)
    print(average_width)
    # print(box)
   
    # move_height = int(average_height - ((box[2]-box[0])//2))
    # move_width = int(average_width - ((box[3]-box[1])//2))
    # move_height = int(new_average_height - average_height) * (int(new_average_height - average_height)<=10)
    # move_width = int(new_average_width - average_width) * (int(new_average_width - average_width)<=10)

    print(int(new_average_height - average_height))
    print(int(new_average_width - average_width))

    #box = [box[0]+move_height,box[1]+move_width,box[2]+move_height,box[3]+move_width]








# for image_path in TEST_IMAGE_PATHS:
#   image = Image.open(image_path)
#   width, height = image.size
#   #Size 1920*945
  
#   start = timer()
#   # the array based representation of the image will be used later in order to prepare the
#   # result image with boxes and labels on it.
#   image_np = load_image_into_numpy_array(image)
#   # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#   image_np_expanded = np.expand_dims(image_np, axis=0)
#   # Actual detection.
#   output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

#   end = timer()
#   print("Elapsed time for 1 image detection: ", end - start)
  
#   boxes_array = output_dict['detection_boxes'].copy()
#   print(boxes_array.shape)

#   #convert pil img to cv2 img
#   pilimg = image.convert('RGB') 
#   cvimg = np.array(pilimg) 
#   cvimg = cvimg[:, :, ::-1].copy()

#   for box in boxes_array:
#     for pos in range(len(box)):
#       if pos%2 == 0:
#         box[pos] *= height
#       else:
#         box[pos] *= width

#     box = box.astype(int)
#     if box[2]-box[0] <= 0 or box[3]-box[1] <= 0:
#       continue
#     else:
#       print(box)
#       boximg = cvimg[box[0]:box[2],box[1]:box[3]]
#       detect = white_or_blue(boximg)
#       cv2.rectangle(cvimg,(box[1],box[0]),(box[3],box[2]),(0,255,0),3)
      
#       cv2.putText(cvimg,detect,(box[1],box[2]), font, 1,(255,255,255),2,cv2.LINE_AA)
      
# cv2.imshow('image',cvimg)
# cv2.waitKey(0)

