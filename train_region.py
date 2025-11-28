import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from region import Region_Net, RegionFeatureExtractor, RegionDataset


TRAIN_DIR = 'train'
TEST_DIR = 'test'
DATA_DIR = 'region_data'
UPSCALE_SIZE = 224
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class FeatureExtractor:
    def __init__(self):
        self.extractor = RegionFeatureExtractor(static_mode=True)
        os.makedirs(DATA_DIR, exist_ok=True)
    
    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Upscale for better landmark detection
        img = cv2.resize(img, (UPSCALE_SIZE, UPSCALE_SIZE), interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return self.extractor.extract_features(img_rgb)
    
    def extract_from_folder(self, data_dir, split_name):
        print(f"\nExtracting features from: {data_dir}")
        print(f"Upscaling: 48x48 -> {UPSCALE_SIZE}x{UPSCALE_SIZE}")
        
        class_folders = sorted([d for d in os.listdir(data_dir) 
                               if os.path.isdir(os.path.join(data_dir, d))])
        
        print(f"Classes: {class_folders}")
        
        all_features = []
        all_labels = []
        failed = 0
        
        for class_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            print(f"\nProcessing: {class_name} ({len(image_files)} images)")
            
            for img_file in tqdm(image_files, desc=f"  {class_name}"):
                features = self.process_image(os.path.join(class_path, img_file))
                
                if features is not None:
                    all_features.append(features)
                    all_labels.append(class_idx)
                else:
                    failed += 1
        
        total = len(all_features) + failed
        print(f"\n{'='*50}")
        print(f"{split_name} extraction complete:")
        print(f"  Successful: {len(all_features)}")
        print(f"  Failed: {failed}")
        print(f"  Success rate: {100*len(all_features)/total:.1f}%")
        
        if len(all_features) == 0:
            return None, None
        
        features_array = np.array(all_features, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.int64)
        
        np.save(os.path.join(DATA_DIR, f'{split_name}_features.npy'), features_array)
        np.save(os.path.join(DATA_DIR, f'{split_name}_labels.npy'), labels_array)
        
        print(f"\nClass distribution:")
        for i, emotion in enumerate(EMOTIONS):
            print(f"  {emotion}: {np.sum(labels_array == i)}")
        
        return features_array, labels_array
    
    def close(self):
        self.extractor.close()


def load_data(split):
    features_path = os.path.join(DATA_DIR, f'{split}_features.npy')
    labels_path = os.path.join(DATA_DIR, f'{split}_labels.npy')
    
    if os.path.exists(features_path) and os.path.exists(labels_path):
        return np.load(features_path), np.load(labels_path)
    return None, None


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\nLoading features...")
    train_features, train_labels = load_data('train')
    test_features, test_labels = load_data('test')
    
    if train_features is None:
        print("ERROR: No training data. Run extraction first.")
        return None
    
    print(f"Train: {len(train_labels)} samples")
    print(f"Test: {len(test_labels)} samples")
    print(f"Features: {train_features.shape[1]}")
    
    train_loader = DataLoader(RegionDataset(train_features, train_labels), 
                             batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(RegionDataset(test_features, test_labels), 
                            batch_size=BATCH_SIZE, shuffle=False)
    
    model = Region_Net(input_size=train_features.shape[1]).to(device)
    
    # Class weights
    counts = np.bincount(train_labels, minlength=7).astype(np.float32)
    counts[counts == 0] = 1
    weights = torch.FloatTensor((1.0 / counts) / (1.0 / counts).sum() * 7).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_acc = 0.0
    patience = 0
    
    print("\n" + "="*50)
    print("TRAINING REGION MODEL")
    print("="*50)
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(features)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loss_sum += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        
        scheduler.step()
        train_losses.append(loss_sum / len(train_loader))
        train_accs.append(100 * correct / total)
        
        # Validate
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                out = model(features)
                loss_sum += criterion(out, labels).item()
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
        
        val_losses.append(loss_sum / len(test_loader))
        val_accs.append(100 * correct / total)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train: {train_accs[-1]:.1f}% | Val: {val_accs[-1]:.1f}%")
        
        if val_accs[-1] > best_acc:
            best_acc = val_accs[-1]
            patience = 0
            torch.save(model.state_dict(), 'emotion_region_best.pth')
            print(f"  -> Saved best model ({best_acc:.1f}%)")
        else:
            patience += 1
            if patience >= 25:
                print("Early stopping")
                break
    
    print(f"\nBest accuracy: {best_acc:.1f}%")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Val')
    ax1.set_title('Loss')
    ax1.legend()
    ax2.plot(train_accs, label='Train')
    ax2.plot(val_accs, label='Val')
    ax2.set_title('Accuracy')
    ax2.legend()
    plt.savefig('emotion_region_training_history.png', dpi=150)
    plt.close()
    
    # Confusion matrix
    model.load_state_dict(torch.load('emotion_region_best.pth'))
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            preds.extend(model(features.to(device)).argmax(1).cpu().numpy())
            labels_all.extend(labels.numpy())
    
    print("\n" + classification_report(labels_all, preds, target_names=EMOTIONS))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(labels_all, preds), annot=True, fmt='d', 
                cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('emotion_region_confusion_matrix.png', dpi=150)
    plt.close()
    
    return model


if __name__ == '__main__':
    print("="*50)
    print("STEP 1: EXTRACTING FEATURES")
    print("="*50)
    
    extractor = FeatureExtractor()
    extractor.extract_from_folder(TRAIN_DIR, 'train')
    extractor.extract_from_folder(TEST_DIR, 'test')
    extractor.close()
    
    print("\n" + "="*50)
    print("STEP 2: TRAINING MODEL")
    print("="*50)
    
    train_model()