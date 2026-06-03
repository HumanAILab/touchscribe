#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 31st December 2017 - new year eve :)
# ----------------------------------------------

from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import itemfreq
from utils.color_recognition.src.color_recognition_api import knn_classifier as knn_classifier
# from color_recognition_api import knn_classifier as knn_classifier


def color_histogram_of_test_image(test_src_image):

    # load the image
    image = test_src_image
    # cv2.imwrite('image.png', test_src_image)
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    red = 0
    green=0
    blue=0

    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
            # print('blue', blue)
        elif counter == 2:
            green = str(elem)
            # print('green', green)
        elif counter == 3:
            red = str(elem)
            # print('red', red)
            feature_data = red + ',' + green + ',' + blue
            # print(feature_data)    
    with open('test.data', 'w') as myfile:
        myfile.write(feature_data)
    return [int(red), int(green), int(blue)]

import os
import cv2
import numpy as np

def color_histogram_of_training_image(img_name):
    # Ensure img_name is an absolute path
    img_name = os.path.abspath(img_name)

    # Detect image color based on filename
    colors = ['red', 'yellow', 'green', 'orange', 'white', 'black', 'blue', 'violet']
    data_source = next((color for color in colors if color in img_name), None)

    if not data_source:
        print(f"Warning: Could not determine color for {img_name}")
        return

    # Load the image
    image = cv2.imread(img_name)
    if image is None:
        print(f"Error: Could not read image {img_name}")
        return

    chans = cv2.split(image)
    features = []
    feature_data = ''

    # Extract color histogram data
    color_order = ('b', 'g', 'r')
    peak_values = []
    
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)
        peak_values.append(str(np.argmax(hist)))

    if len(peak_values) == 3:
        feature_data = ','.join(peak_values[::-1])  # Reverse order to RGB format

    print("feature_data:", feature_data)
    print("data_source:", data_source)

    # Save training data to a consistent directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    training_data_path = os.path.join(project_root, 'training.data')

    with open(training_data_path, 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')

    print(f"Training data saved to: {training_data_path}")


# def color_histogram_of_training_image(img_name):

#     # detect image color by using image file name to label training data
#     if 'red' in img_name:
#         data_source = 'red'
#     elif 'yellow' in img_name:
#         data_source = 'yellow'
#     elif 'green' in img_name:
#         data_source = 'green'
#     elif 'orange' in img_name:
#         data_source = 'orange'
#     elif 'white' in img_name:
#         data_source = 'white'
#     elif 'black' in img_name:
#         data_source = 'black'
#     elif 'blue' in img_name:
#         data_source = 'blue'
#     elif 'violet' in img_name:
#         data_source = 'violet'

#     # load the image
#     image = cv2.imread(img_name)

#     chans = cv2.split(image)
#     colors = ('b', 'g', 'r')
#     features = []
#     feature_data = ''
#     counter = 0
#     for (chan, color) in zip(chans, colors):
#         counter = counter + 1

#         hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
#         features.extend(hist)

#         # find the peak pixel values for R, G, and B
#         elem = np.argmax(hist)

#         if counter == 1:
#             blue = str(elem)
#         elif counter == 2:
#             green = str(elem)
#         elif counter == 3:
#             red = str(elem)
#             feature_data = red + ',' + green + ',' + blue

#     print("feature_data", feature_data)
#     print("data_source", data_source)
#     with open('training.data', 'a') as myfile:
#         myfile.write(feature_data + ',' + data_source + '\n')


def training():
    print("color is training....")

    # Move up one directory level to get the correct path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training_dataset'))

    colors = ['red', 'yellow', 'green', 'orange', 'white', 'black', 'blue']

    for color in colors:
        color_dir = os.path.join(base_dir, color)
        if os.path.exists(color_dir):
            for f in os.listdir(color_dir):
                color_histogram_of_training_image(os.path.join(color_dir, f))
                print(os.path.join(color_dir, f))
        else:
            print(f"Warning: Directory {color_dir} does not exist.")


# def training():
#     print("color is training....")
#     # red color training images
#     for f in os.listdir('./training_dataset/red'):
#         color_histogram_of_training_image('./training_dataset/red/' + f)

#     # yellow color training images
#     for f in os.listdir('./training_dataset/yellow'):
#         color_histogram_of_training_image('./training_dataset/yellow/' + f)

#     # green color training images
#     for f in os.listdir('./training_dataset/green'):
#         color_histogram_of_training_image('./training_dataset/green/' + f)

#     # orange color training images
#     for f in os.listdir('./training_dataset/orange'):
#         color_histogram_of_training_image('./training_dataset/orange/' + f)

#     # white color training images
#     for f in os.listdir('./training_dataset/white'):
#         color_histogram_of_training_image('./training_dataset/white/' + f)

#     # black color training images
#     for f in os.listdir('./training_dataset/black'):
#         color_histogram_of_training_image('./training_dataset/black/' + f)

#     # blue color training images
#     for f in os.listdir('./training_dataset/blue'):
#         color_histogram_of_training_image('./training_dataset/blue/' + f)		
