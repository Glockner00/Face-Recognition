import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # find the path to the file
IMAGE_DIR = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer.create() # facial recognizer, LPBH, TODO: try another recognizer.

current_id = 0
label_ids={}
y_labels = [] #numbers related to the labels
x_train = [] #numbers of the pixel values

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()      #can replace path with 'root'      
            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
            id_ = label_ids[label]
            
            pil_image = Image.open(path).convert("L") # saving the image from the path and converting to grayscale.
            image_array = np.array(pil_image, "uint8") # image(pixel values) into numpy array, type uint8
            print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5) #detector 
            for x,y,w,h in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle", "wb") as f: #save the label ids
    pickle.dump(label_ids, f)

# training data in x_train, convert the y_labels into np arrays.
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")