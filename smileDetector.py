# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 02:30:11 2020

@author: Abdul

"""

# Face Recognition

import numpy as np
import cv2

# loading cascade for face and smile detetcion
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Building a function that takes frame as input and detect the face and smile and draws retangle around it
def detect(gray, frame):
    # first we'll detect face and in that region we will detect smile because this helps in reducing inaccuracy/false positive
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # detecting smile
        smile = smile_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in smile:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('frame', canvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()