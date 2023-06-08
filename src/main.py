"""
Main program. Collect images and run myTrainer.py before running main.
Works with multiple faces in a single frame.
quit with q.
"""
import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("trainer.yml")

score = {}
first_labels = {}
with open("labels.pickle", "rb") as f: #save the label ids
    first_labels = pickle.load(f)
    labels = {v:k for k, v in first_labels.items()} #inverting the labels.

for k, v in first_labels.items():
    score[k] = []

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read() #capture frame by frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # a gray frame (a must for opencv facial recognition to work)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) # (coordinates for my face (x,y,w,h))
    for (x,y,w,h) in faces:
        #---------saving the last frame as an image----------#
        roi_gray = gray[y:y+h, x:x+w] #region of interest gray
        id_, conf  = recognizer.predict(roi_gray) # label, confidence
        print(int(conf), " : ", labels[id_].replace("-", " ").title())
        if conf>= 30:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_].replace("-", " ")
            color = (255,255,255)
            cv2.putText(frame, name.title(), (x,y+h+25), font, 0.8 , color, 2, cv2.LINE_AA)
            score[labels[id_]].append(conf)

        img_item = "my_image.png"
        cv2.imwrite(img_item, roi_gray)

        #--------Surrounding my face with a rectangle--------# 
        roi_color = frame[y:y+h, x:x+w] #region of interest color
        end_pos_x = x+w
        end_pos_y= y+h 
        cv2.rectangle(frame, (x,y), (end_pos_x, end_pos_y), color=(0,0,255), thickness=2)

    cv2.imshow('facecam', frame) # Display a frame (in color)
    if cv2.waitKey(20) & 0xFF == ord('q'): # close window with 'q'
        break

cap.release()
cv2.destroyAllWindows()

average_score = {}

for k, v in first_labels.items():
    average_score[k] = 0

for k, v in score.items():
    if len(v)>0:
        average_score[k] = sum(v)/len(v)
    else:
        average_score[k] = 0.0

for k, v in average_score.items():
    print(k.replace("-", " ").title(), ": ", v)