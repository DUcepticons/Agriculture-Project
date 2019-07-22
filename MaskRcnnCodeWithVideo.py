# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 17:01:11 2019

@author: Riad
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:19:50 2019

@author: Riad
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:37:45 2019

@author: Riad
"""

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import visualize
import numpy as np
import cv2
import colorsys
# define 81 classes that the coco model knowns about
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
 
hsv = [(i / len(class_names), 1, 1.0) for i in range(len(class_names))]
COLORS =lambda c: list(map( colorsys.hsv_to_rgb(*c), hsv))

# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80
 
# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
# load photograph
cap = cv2.VideoCapture(0)

camera = cv2.VideoCapture(0)

while(True):
    ret, image = cap.read()
    #img = load_img('airplane2.jpg')
    img = img_to_array(image)
    #image=cv2.imread('airplane2.jpg')
    # make prediction
    result = rcnn.detect([img], verbose=0)
    r=result[0]
    #print((r["rois"].shape[0]))
    for i in range(0, len(r["scores"])):        
        (startY, startX, endY, endX) = r["rois"][i]
        classID = r["class_ids"][i]
        label = class_names[classID]
        score = r["scores"][i]
        # show the output image
        w,h, channels = img.shape
        print(w,h)
        w,h, channels = image.shape
        print(w,h)
        #image=cv2.resize(image,(w,h))
        
        #print(image.shape[:1])
        cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
        text = "{}: {:.3f}".format(label, score)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
        		0.6, (0,255,0), 2)
    cv2.imshow("Output", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        
        break

cv2.destroyAllWindows()
# get dictionary for first prediction
'''r = results[0]
# show photo with bounding boxes, masks, class labels and scores
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])'''