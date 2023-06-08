"""
Untar Labelled Faces in the wild dataset : http://vis-www.cs.umass.edu/lfw/
"""

# Import standard dependencies
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Import tensorflow dependencies - Functional API
from tensorflow import keras
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from keras.models import Model 
import tensorflow as tf

# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Make directories

#os.makedirs(POS_PATH)
#os.makedirs(NEG_PATH)
#os.makedirs(ANC_PATH)

# Uncompress targz Labelled Faces in the Wild Dataset
# !tar -xf lfw.tgz

# Move LFW images to data/negative, no folders.

#for directory in os.listdir('lfw'):
#    for file in os.listdir(os.path.join('lfw', directory)):
#        EX_PATH = os.path.join('lfw', directory, file)
#       NEW_PATH = os.path.join(NEG_PATH, file)
#        os.replace(EX_PATH, NEW_PATH)

cap = cv2.VideoCapture(0)                # establish a connection to the webcam
while cap.isOpened():                    
    ret, frame = cap.read()
    cv2.imshow('Image Collector', frame) # show image 
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()