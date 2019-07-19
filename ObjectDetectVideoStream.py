# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:21:09 2019

@author: Riad
"""

import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]
model = tf.keras.models.load_model("64x3-CNN (2).model")

def prepare(file):
    IMG_SIZE = 50  # 50 in txt-based
    
    new_array = cv2.resize(file, (IMG_SIZE, IMG_SIZE))
    new_array=new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    new_array=new_array.astype('float32')
    new_array /=255
    return new_array

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    try:
        prediction = model.predict([prepare(gray)])
        dog_per=int((1-prediction[0][0])*100)
        cat_per=int(prediction[0][0]*100)
        #print("Cat: ",cat_per,"%")# will be a list in a list.
        #print("Dog: ",dog_per,"%")'
        print(dog_per)
        if dog_per > 65 :
            name="Dog: {}%".format(dog_per)
        else:
            name=""
        '''if cat_per > 65 :
            name="Cat: {}%".format(cat_per)'''
    except Exception as e:
        
        print(e)

    #cnts=imutils.contours(cnts)
	#gray = cv2.GaussianBlur(gray,(5, 5),0)
    # Display the resulting frame
    canny = cv2.Canny(gray,80,240,3)
    
    
    if ret==True:
        #grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Canny
        canny = cv2.Canny(frame,80,240,3)

        #contours
        canny2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0,len(contours)):
            #approximate the contour with accuracy proportional to
            #the contour perimeter
            approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.02,True)

            #Skip small or non-convex objects
            if(abs(cv2.contourArea(contours[i]))<100 or not(cv2.isContourConvex(approx))):
                continue

            #triangle
            if(len(approx) == 3):
                x,y,w,h = cv2.boundingRect(contours[i])
                
            else:
                #detect and label circle
                area = cv2.contourArea(contours[i])
                x,y,w,h = cv2.boundingRect(contours[i])
                radius = w/2
                
            if len(contours) > 0:
                
            
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if radius > 1:
                    if "Dog" in name:
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.circle(frame, (int(x), int(y)), int(radius),(0,0,255) , 2)
                        #cv2.putText(frame,(int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,225,0),2)
                        cv2.putText(frame, name, (int(x+ 6),int( y - 6)), font, 1, (255, 255, 255))
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    
                    cv2.imshow('frame',frame)
                  
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




