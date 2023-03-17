import json
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # get rid of tensorflow errors
import tkinter as tk
from tkinter import filedialog
# All imports below are published on pypi.org
# regex 2021.11.10
import regex
# Pillow 9.0.0
from PIL import Image
# requests 2.27.1
import requests
# aiml 0.9.2

# pandas 1.3.5
import pandas as pd
# simpful 2.5.1

# numpy 1.22.3
import numpy as np

import tensorflow as tensor

def cleanNames(name):
    symbols = ['/', '-']  # list of symbols to remove
    for symbol in symbols:
        name = name.replace(symbol, ' ')
    return name

#imgPath ="./Photos/Charles-Michel.jpg"
root = tk.Tk()
root.filename = filedialog.askopenfilename(initialdir="test_data/",
                                            title="what is this image?",
                                            filetypes=(("JPG files", "*.jpg"),("All Files", "*.*")))
root.destroy()
imgPath =  root.filename
#get image name without extension
imgName = imgPath.split("/")[-1]
imgName = cleanNames(imgName.split(".")[0])

if(imgPath != ""):
    # CNN model
    # Load EuropeanWodnerModel.h5 model
    model= tensor.keras.models.load_model("EuropeanModel.h5")

    # Load selected image into correct format for model
    img = tensor.keras.utils.load_img(imgPath, target_size = (180,180))
    imgArray = tensor.keras.utils.img_to_array(img)
    imgArray = tensor.expand_dims(imgArray, axis = 0)

    # Use model to predict score
    output = model.predict(imgArray)
    score = tensor.nn.softmax(output[0])

    # Output predicted class with confidence percentage
    class_names = ['Flag', 'Human']
    print("this is {} - {} class with confidence percentage {} %".format(imgName, class_names[np.argmax(score)], 100 * np.max(score)))
else:
    print("No image selected.")
