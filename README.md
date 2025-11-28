# Emotion Recognition — CS4100 Final Project

## Quick Start (Mac)

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
