"""
Basic chatbot design --- for your own modifications
"""
#######################################################
# Initialise Wikipedia agent
#######################################################
import wikipedia
import inference
# Import matplotlip to show images
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#######################################################
# Initialise weather agent
#######################################################
import json, requests
#insert your personal OpenWeathermap API key here if you have one, and want to use this feature
APIkey = "5403a1e0442ce1dd18cb1bf7c40e776f" 

#######################################################
#  Initialise AIML agent
#######################################################
import aiml

#######################################################
#  Initialise TensorFlow CNN model
#######################################################

import json
import csv
import os
import random

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

# predict function for CNN model
def predict(imgPath):
    img = tensor.keras.utils.load_img(imgPath, target_size = (180,180))
    imgArray = tensor.keras.utils.img_to_array(img)
    imgArray = tensor.expand_dims(imgArray, axis = 0)

    # Use model to predict score
    output = model.predict(imgArray)
    score = tensor.nn.softmax(output[0])
    class_names = ['Flag', 'Human']
    return class_names[np.argmax(score)], 100 * np.max(score)

#select random file from directory
def select_random_file(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Filter out directories and hidden files
    files = [f for f in files if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.')]

    # Select a random file from the list
    if len(files) > 0:
        random_file = random.choice(files)
        return os.path.join(directory, random_file)
    else:
        return None
# Load EuropeanModel.h5 model
model= tensor.keras.models.load_model("EuropeanModel.h5")

human_path = []
flag_path = []
#######################################################
#  Initialise CSV, regular expression and math libraries
#######################################################
import csv
import re
import math


# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="mybot-basic.xml")
#######################################################
# Welcome user
#######################################################
print("Welcome to this chat bot. Please feel free to ask questions from me!")
#######################################################
# Main loop
#######################################################
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
       
        elif cmd == 3:
            inference.inference_remember(params[1])
            

        elif cmd == 4:
            inference.inference_check(params[1])
        
        elif cmd == 5:

            inference.inference_kb()  

        elif cmd == 6:
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
                class_name, score = predict(imgPath)
                print("this is {} - {} class with confidence percentage {} %".format(imgName, class_name, score))
            else:
                print("No image selected.")

        elif cmd == 7:

            # select rundom file from Photos folder and return its path
            path = './Photos/Human'

            # get all files in the directory
            imgPath =  select_random_file(path)
            #get image name without extension
            imgName = imgPath.split("/")[-1]
            imgName = cleanNames(imgName.split(".")[0])
            human_path.append(imgPath)

            if(imgPath != ""):
                # Open the image file
                image = Image.open(imgPath)

                # Display the image
                image.show()
            else:
                print("No image selected.")
    
        elif cmd == 8:
            imgPath =  human_path[-1]
            #get image name without extension
            imgName = imgPath.split("/")[-1]
            imgName = cleanNames(imgName.split(".")[0])
            if(imgPath != ""):
                class_name, score = predict(imgPath)
                print("this is {} EU President - {} class with confidence percentage {} %".format(imgName, class_name, score))
            else:
                print("No image selected.")
    
        elif cmd == 9:
                        # select rundom file from Photos folder and return its path
            path = './Photos/flag'

            # get all files in the directory
            imgPath =  select_random_file(path)
            #get image name without extension
            imgName = imgPath.split("/")[-1]
            imgName = cleanNames(imgName.split(".")[0])
            flag_path.append(imgPath)

            if(imgPath != ""):
                # Open the image file
                image = Image.open(imgPath)

                # Display the image
                image.show()
            else:
                print("No image selected.")
  
        elif cmd == 10:
            imgPath =  flag_path[-1]
            #get image name without extension
            imgName = imgPath.split("/")[-1]
            imgName = cleanNames(imgName.split(".")[0])
            if(imgPath != ""):
                class_name, score = predict(imgPath)
                print("this is {} flag - {} class with confidence percentage {} %".format(imgName, class_name, score))
            else:
                print("No image selected.")




    else:
        print(answer)