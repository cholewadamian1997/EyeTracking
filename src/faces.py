#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 20:28:54 2021

@author: damian
"""

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(G, scaleFactor=1.5, minNeighbors=5)
    
    for (x1, y1, w1, h1) in faces:
        cv2.rectangle(frame, (x1,y1),(x1+w1,y1+h1),(0,255,0),2)

        eyes = eye_cascade.detectMultiScale(frame)
        for (x2, y2, w2, h2) in eyes:
            cv2.rectangle(frame, (x2,y2),(x2+w2,y2+h2),(255,0,0),1)

    
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()


