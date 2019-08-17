# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 23:15:15 2019

@author: Riad
"""
import os
import cv2
from numpy import expand_dims
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import save_img

# load the image
path = 'C:\\Users\\Riad\\Desktop\\Eggplant'
files = os.listdir(path)
for file in files:
    
    img = cv2.imread(os.path.join(path,file))
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    #datagen = ImageDataGenerator(width_shift_range=[-200,200])
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
    height_shift_range=0.1,zoom_range=0.1,channel_shift_range = 10, horizontal_flip=True)
    save_here = "C:\\Users\\Riad\\Desktop\\Eggplant_new"
    
    for x, val in zip(datagen.flow(samples,                   
            save_to_dir=save_here,    
             save_prefix='aug',        
            save_format='jpg'),range(10)):
        pass

