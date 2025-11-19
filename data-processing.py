import pandas as pd
import numpy as np
import os
import cv2
import librosa
import torch
import keras
import seaborn as sns
import matplotlib
import os
import mediapipe

# this is how we are downloading the data/auth
kaggle.api.authenticate()




# os.environ["KAGGLE_CONFIG_DIR"] = "/Users/theresa/my-api-keys/kaggle.json"

# audio data
kaggle.api.dataset_download_files("ejlok1/cremad",path = ".", unzip=True)


# video/photo data


# processing the data
