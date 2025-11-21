import pandas as pd
import numpy as np
import os
import cv2
import seaborn as sns
import matplotlib
import os

import mediapipe as mp # this is for landmarker stuff
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import convert_rgb
# from ultralytics import YOLO

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    min_detection_confidence=0.1 # lowered threshold to detect face
)


# prepping data
emotions_label = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

all_image_path = []
label_data = []

for emotion_index, emotion in enumerate(emotions_label):
    image_folder_path = f"dataset/images/train/{emotion}"  
    for filename in os.listdir(image_folder_path):
        if filename.lower().endswith(".jpg"):
            #keeping track of the image path 
            image_path = os.path.join(image_folder_path, filename)
            all_image_path.append(image_path)
            label_data.append(emotion_index)

print(f"this is the total number of images:{len(all_image_path)}")

# end result should be all of the features you can draw from mediapipe
# every datapoint when you run an image, google mediapipe 
# will give you features (x,y,z) potential
# and then give you a confidence score
# iteratively feed images in to google mediapipe
# spit out data + tag that on to vector 

# Initialize mediapipe components -- will find faces and extract features
model_path = 'face_landmarker_v2_with_blendshapes.task'

FaceLandmarkerOptions = vision.FaceLandmarkerOptions
FaceLandmarker = vision.FaceLandmarker
base_options = python.BaseOptions(model_asset_path = model_path)

# looking for facial expressions and head poses
options = FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
    )

# FEATURE EXTRACTION
pulled_features = []
valid_labels = []


with FaceLandmarker.create_from_options(options) as landmarker:
    for image_idx, image_info in enumerate(all_image_path):
        rgb_img = convert_rgb(image_info)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        result = landmarker.detect(mp_image) 

        print("amde it past here")

        # no face detected = skip
        if not result:
            print(f"no face detected in {image_info}")
            continue

        features = []
        # get the face landmarks
        
        if result.face_landmarks:
            for landmark in result.face_landmarks[0]:
                features.append(landmark.x)
                features.append(landmark.y)
                features.append(landmark.z)
        else:
            print("nothing in face landmarks")
            break
            
        # facial expressions (we will know from media pipe)
        for face_expressions in result.face_blendshapes[0]:
            features.append(face_expressions.score)

        # head poses
        given_matrix = result.facial_transformation_matrixes[0]
        # add all values
        for row in given_matrix:
            for col in row:
                features.append(col)

        # add the features + labels to some list
        pulled_features.append(features)
        #valid_labels.append(label)
    

