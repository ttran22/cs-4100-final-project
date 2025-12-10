# Emotion Recognition — CS4100 Final Project

## Quick Start ResNet (Mac)

### 1. Run setup script
```bash
sh setup_mac.sh
```

### 2. Activate environment
```bash
source ml/bin/activate
```

### 3. Preprocess faces
```bash
python preprocess_faces.py
```

### 4. Train the ResNet18 model
```bash
python train_resnet.py
```

## Output Files
- `resnet_emotion_best.pth` — best model weights
- `resnet_emotion_training_history.png`
- `resnet_emotion_confusion_matrix.png`

## Notes
- Works on M1, M2, M3, M4, M5 chips using PyTorch MPS.
- Auto-detects GPU → CPU fallback also supported.
- CutMix + MixUp + strong aug produce best accuracy.

## FFN and CNN

### 1. Make sure requirements.txt are installed
```bash
pip install -r requirements.txt
```

### 2. Run emotion.py
Note that you can edit which directory is used as the training and test sets for training and evaluations on the different data. In file emotion.py on lines 46-47.

### Output Files
- `emotion_ffn_best.pth`
- `emotion_ffn_confusion_matrix.png`
- `emotion_ffn_examples.png`
- `emotion_ffn_training_history.png`
- `emotion_cnn_best.pth`
- `emotion_cnn_confusion_matrix.png`
- `emotion_cnn_examples.png`
- `emotion_cnn_training_history.png`

## Region

### 1. Make sure requirements.txt are installed
```bash
pip install -r requirements.txt
```

### 2. Run train_region.py
This will train and evaluate on the test and train directories. 

### Output Files
- `emotion_region_confusion_matrix.png`
- `emotion_region_training_history.png`

## Realtime Detector (Webcam)

### 1. Make sure requirements.txt are installed
```bash
pip install -r requirements.txt
```

### 2. Run realtime.py
This will display a webcam and instructions are in the terminal.