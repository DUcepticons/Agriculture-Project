# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:42:09 2019

@author: Riad
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,SeparableConv2D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle
DATADIR = "E:\DeepLearning\CatDog\PetImages"

CATEGORIES = ["Dog", "Cat"]


training_data = []
IMG_SIZE = 70

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

def build_model(S):
    '''S=list(S)
    S.insert(0,32)
    S=tuple(S)'''
    #S=np.expand_dims(S, axis=0)
    image = Input(S)
    #image = np.expand_dims(image, axis=0)
    val = SeparableConv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(image)
    val = SeparableConv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(val)
    val = MaxPooling2D((2,2), name='pool1')(val)
    
    val = SeparableConv2D(128,(3,3), activation='relu', padding='same', name='Conv2_1')(val)
    val = BatchNormalization(name='bn1')(val)
    val = SeparableConv2D(128,(3,3), activation='relu', padding='same', name='Conv2_2')(val)
    val = BatchNormalization(name='bn2')(val)
    val = MaxPooling2D((2,2), name='pool2')(val)
    
    val = SeparableConv2D(256,(3,3), activation='relu', padding='same', name='Conv3_1')(val)
    val = BatchNormalization(name='bn3')(val)
    val = SeparableConv2D(256,(3,3), activation='relu', padding='same', name='Conv3_2')(val)
    val = BatchNormalization(name='bn4')(val)
    val = SeparableConv2D(256,(3,3), activation='relu', padding='same', name='Conv3_3')(val)
    val = MaxPooling2D((2,2), name='pool3')(val)
    
    val = SeparableConv2D(512,(3,3), activation='relu', padding='same', name='Conv4_1')(val)
    val = BatchNormalization(name='bn5')(val)
    val = SeparableConv2D(512,(3,3), activation='relu', padding='same', name='Conv4_2')(val)
    val = BatchNormalization(name='bn6')(val)
    val = SeparableConv2D(512,(3,3), activation='relu', padding='same', name='Conv4_3')(val)
    val = MaxPooling2D((2,2), name='pool4')(val)
    
    val = Flatten()(val)
    val = Dropout(0.7, name='dropout1')(val)
    val = Dense(1024, activation='relu', name='fc1')(val)
    val = Dense(2, activation='softmax',name='fc3')(val)
    
    model = Model(inputs=image,outputs=val)
    return model
print("Building_Model.....")
model = build_model(X.shape[1:])
print("Compiling Model...")
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=5, validation_split=0.3)
model.save('new_model.model')
print("Model Saved")