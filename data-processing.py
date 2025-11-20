import pandas as pd
import numpy as np
import os
import cv2
import torch
import keras
import seaborn as sns
import matplotlib
import os
import mediapipe as mp

from mediapipe.tasks.python import vision
from skimage.io import imread
from skimage.transform import resize
from ultralytics import YOLO

# prepping data
emotions_label = ["angry", "disgust", "fear", "happy", "neutral", "sad", "suprise"]
image_data = []

# for emotion_index, emotion in enumerate(emotions_label):
#     image_folder_path = f"/cs-4100-final-project/dataset/images/train/{emotion}"  
#     for filename in os.listdir(image_folder_path):
#         if filename.lower().endswith(".jpg"):
#             image_path = os.path.join(image_folder_path, filename)
#             image = imread(image_path)
#             image_data.append(image)

# end result should be all of the features you can draw from mediapipe
# every datapoint when you run an image, google mediapipe 
# will give you features (x,y,z) potential
# and then give you a confidence score
# iteratively feed images in to google mediapipe
# spit out data + tag that on to vector 


for emotion_index, emotion in enumerate(emotions_label):
    image_folder_path = f"/cs-4100-final-project/dataset/images/train/{emotion}"  
    for filename in os.listdir(image_folder_path):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(image_folder_path, filename)
            image_paths.append(image_path)
            emotion_labels.append(emotion_index)  # Store the emotion label

# Initialize mediapipe components
model_path = 'face_landmarker_v2_with_blendshapes.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1)


mp_group = []
with FaceLandmarker.create_from_options(options) as landmarker:
    for image in image_data:
        mp_image = mp.Image.create_from_file(image)
        mp_group.append(mp_image)

for mp_image in mp_group:
    face_landmarker_result = landmarker.detect(mp_image)



















