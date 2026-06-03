#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 8th July 2018 - before Google inside look 2018 :)
# -------------------------------------------------------------------------

import cv2
from utils.color_recognition.src.color_recognition_api import color_histogram_feature_extraction
from utils.color_recognition.src.color_recognition_api import knn_classifier
# from color_recognition_api import color_histogram_feature_extraction
# from color_recognition_api import knn_classifier
import os
import os.path
import sys

# read the test image
# try:
#     source_image = cv2.imread(sys.argv[1])
# except:
#     source_image = cv2.imread('black_cat.jpg')
# prediction = 'n.a.'

# checking whether the training data is ready
PATH = 'utils/color_recognition/src/training.data'
# PATH = './training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print ('training data is ready, classifier is loading...')
else:
    print ('training data is being created...')
    open(PATH, 'w')
    color_histogram_feature_extraction.training()
    print ('training data is ready, classifier is loading...')

# get the prediction
# open('utils/color_recognition/src/training.data', 'w')
# color_histogram_feature_extraction.training()

def get_color(image_cv2):
    cv2.imwrite('crop.png', image_cv2)
    test = color_histogram_feature_extraction.color_histogram_of_test_image(image_cv2)
    # print("feature", test)
    prediction = knn_classifier.main('utils/color_recognition/src/training.data', 'utils/color_recognition/src/test.data', test)
    # prediction = knn_classifier.main('training.data', 'test.data')

    # print('Detected color is:', prediction)
    return prediction


def get_rgb(image_cv2):
    cv2.imwrite('crop_rgb.png', image_cv2)
    test = color_histogram_feature_extraction.color_histogram_of_test_image(image_cv2)
    return test

# source_image = cv2.imread('/Users/rueichechang/Projects/worldscribe/frame.png')
# color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
# prediction = knn_classifier.main('training.data', 'test.data')
# print('Detected color is:', prediction)
# print(get_color(source_image))
# cv2.putText(
#     source_image,
#     'Prediction: ' + prediction,
#     (15, 45),
#     cv2.FONT_HERSHEY_PLAIN,
#     3,
#     200,
#     )

# # Display the resulting frame
# cv2.imshow('color classifier', source_image)
# cv2.waitKey(0)		
