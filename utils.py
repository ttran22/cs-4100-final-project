import cv2
import mediapipe as mp


def convert_rgb(image_path):
    #current grey image

    original_image = cv2.imread(image_path)
    resized_image = cv2.resize(original_image, (256, 256))

    # checks channels
    if len(original_image.shape) != 3:
        #convert to rgb
        new_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
    else:
        new_image = resized_image

    return new_image