from __future__ import division, absolute_import
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from os.path import isfile, join
import random
import sys
import tensorflow as tf
import os
import glob
import cv2
import sys

from model import EMR

# epoch =
# epoch = int(input('input epoch : '))

# prevents appearance of tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# prevents opencl usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'scared']


def get_files():
    files = glob.glob("./input_picture/*")
    input_data = []

    data = files[:int(len(files))]
    for item in data:
        image = cv2.imread(item)  # open image
        newimg = format_image(image)

        if(len(newimg) == 1):
            continue
        input_data.append(newimg)

    return input_data


def format_image(image):
    """
    Function to format frame
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
            # determine whether the image is color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Image read from buffer
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

    if not len(faces) > 0:
        return [0]

    # initialize the first face as having maximum area, then find the one with max_area
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face

    # extract ROI of face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    try:
        # resize the image so that it can be passed to the neural network
        image = cv2.resize(
            image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("----->Problem during resize")
        return None

#     return image
    output = np.array(image)
    output = np.reshape(output, (48, 48, 1))

    return output


# # Create Training set and Testing Set
data = get_files()

network = EMR()
network.build_network()

for item in data:
    if(len(item) == 48):
        try:
            result = network.predict([item])
        except:
            continue
    else:
        result = None

if result is not None:
    maxindex = np.argmax(result[0])
    print(EMOTIONS[maxindex])
