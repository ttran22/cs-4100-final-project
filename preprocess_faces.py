"""
Detect and crop faces from train/ and test/ using MediaPipe Face Detection.
Save 224x224 RGB crops into faces/train/<emotion>/ and faces/test/<emotion>/.
"""

import os
import cv2
from tqdm import tqdm
import mediapipe as mp

#AI was used throughout the code. 

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

INPUT_TRAIN_DIR = "train"
INPUT_TEST_DIR = "test"

OUTPUT_ROOT = "faces"
OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_ROOT, "train")
OUTPUT_TEST_DIR = os.path.join(OUTPUT_ROOT, "test")

FACE_SIZE = 224  # target size for ResNet


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_output_dirs():
    for split_dir in [OUTPUT_TRAIN_DIR, OUTPUT_TEST_DIR]:
        for emotion in EMOTIONS:
            ensure_dir(os.path.join(split_dir, emotion))


def detect_and_crop_faces(image_path: str, detector):
    """
    Detect faces in an image and return a list of 224x224 BGR crops.
    Typically we use only the first face.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.process(img_rgb)

    if not results.detections:
        return []

    h, w, _ = img.shape
    crops = []

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box

        x_min = max(int(bbox.xmin * w), 0)
        y_min = max(int(bbox.ymin * h), 0)
        x_max = min(int((bbox.xmin + bbox.width) * w), w - 1)
        y_max = min(int((bbox.ymin + bbox.height) * h), h - 1)

        margin = int(0.1 * (x_max - x_min))
        x_min = max(x_min - margin, 0)
        y_min = max(y_min - margin, 0)
        x_max = min(x_max + margin, w - 1)
        y_max = min(y_max + margin, h - 1)

        face = img[y_min:y_max, x_min:x_max]
        if face.size == 0:
            continue

        face = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
        crops.append(face)

    return crops


def process_split(input_root: str, output_root: str, detector):
    print(f"\n=== Processing split: {input_root} -> {output_root} ===")

    total_images = 0
    saved = 0
    no_face = 0

    for emotion in EMOTIONS:
        in_dir = os.path.join(input_root, emotion)
        out_dir = os.path.join(output_root, emotion)

        if not os.path.isdir(in_dir):
            continue

        files = [
            f for f in os.listdir(in_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        print(f"{emotion}: {len(files)} images")
        for fname in tqdm(files, desc=f"  {emotion}"):
            total_images += 1
            img_path = os.path.join(in_dir, fname)
            crops = detect_and_crop_faces(img_path, detector)

            if not crops:
                # Fallback: use full image resized to 224x224
                img = cv2.imread(img_path)
                if img is None:
                    no_face += 1
                    continue
                face = cv2.resize(img, (FACE_SIZE, FACE_SIZE),
                                  interpolation=cv2.INTER_AREA)
            else:
                face = crops[0]

            out_name = os.path.splitext(fname)[0] + "_face.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, face)
            saved += 1

    print(
        f"Done: total={total_images}, saved={saved}, no_face={no_face}"
    )


def main():
    ensure_dir(OUTPUT_ROOT)
    build_output_dirs()

    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as detector:
        process_split(INPUT_TRAIN_DIR, OUTPUT_TRAIN_DIR, detector)
        process_split(INPUT_TEST_DIR, OUTPUT_TEST_DIR, detector)

    print("\nFace preprocessing complete. Output in 'faces/'.")
    

if __name__ == "__main__":
    main()