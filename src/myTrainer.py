import os
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # find the path to the file
IMAGE_DIR = os.path.join(BASE_DIR, "images")
y_labels = []
x_train = []

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            #can replace path with 'root'
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            print(label, path)
            
            pil_image = Image.open(path).convert("L") # saving the image from the path and converting to grayscale.
            image_array = np.array(pil_image, "uint8") # image(pixel values) into numpy array, type uint8
            print(image_array)