"""
This is a program for collecting data (images) for the cascade classifier trainer.
Enter name as follows: "firstname-lastname"
Look straight into the camera, and do different expressions.
"""
import numpy as np
import cv2
import os

inp = input("Name: ")
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
data_dir = os.path. join(IMAGE_DIR, inp)
count=1

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) 
    
    for (x,y,w,h) in faces:
        img_item = os.path.join(data_dir, f"{int(count/10)}.png")
        if (count%10)==0:
            cv2.imwrite(img_item, frame)
        count+=1
    cv2.imshow('facecam', frame) 

    if cv2.waitKey(20) & 0xFF == ord('q'): 
        break
    elif (count==510):
        break

cap.release()
cv2.destroyAllWindows()