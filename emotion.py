import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
from ffn import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
weights =  [1.5, 3.0, 1.5, 0.8, 1.1, 1.0, 1.2]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

transform = transforms.Compose([                            # Use transforms to convert images to tensors and normalize them
    transforms.Grayscale(num_output_channels=1),            # Ensure grayscale
    transforms.Resize((48, 48)),                            # Ensure size 48x48
    transforms.ToTensor(),                                  # convert images to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])             # Common method for grayscale images
])

# Data augmentation for training (hopefully improves accuracy)
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),            # Ensure grayscale
    transforms.Resize((48, 48)),                            # Ensure size 48x48
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Add random shifts
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

batch_size = 128

trainset = datasets.ImageFolder(root='train', transform=train_transform)
testset = datasets.ImageFolder(root='test', transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Training function
def train_model(model, model_name, num_epochs=50, learning_rate=0.001):
    print(f'Training {model_name}')
    
    model = model.to(device)
    class_weights = torch.tensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                               T_0=10,      # Restart every 10 epochs
                                                               T_mult=1,    # Keep cycle length constant
                                                               eta_min=1e-6 # Minimum learning rate
                                                               )
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    patience = 0
    max_patience = 7
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Flatten for FFN if needed
            if 'FFN' in model_name:
                images = images.view(-1, 2304)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            scheduler.step(epoch + batch_idx / len(trainloader))
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                
                if 'FFN' in model_name:
                    images = images.view(-1, 2304)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(testloader)
        val_epoch_acc = 100 * correct / total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%')
        print(f'  Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%')
        
        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience = 0
            torch.save(model.state_dict(), f'{model_name.lower().replace(" ", "_")}_best.pth')
            print(f'Best model saved! (Val Acc: {best_val_acc:.2f}%)')
        else:
            patience += 1
            if patience >= max_patience:
                print(f"No improvement for {max_patience}")
                break
    
    print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')
    
    return train_losses, train_accs, val_losses, val_accs

# Evaluation function
def evaluate_model(model, model_name):
    print(f'Evaluating {model_name}')
    
    model.eval()
    all_preds = []
    all_labels = []
    
    # Store examples
    correct_examples = {i: None for i in range(7)}
    incorrect_examples = {i: None for i in range(7)}
    
    with torch.no_grad():
        for images, labels in testloader:
            images_orig = images.clone()
            images, labels = images.to(device), labels.to(device)
            
            if 'FFN' in model_name:
                images = images.view(-1, 2304)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Collect examples
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                
                if pred == label and correct_examples[label] is None:
                    correct_examples[label] = (images_orig[i].cpu(), label, pred)
                elif pred != label and incorrect_examples[label] is None:
                    incorrect_examples[label] = (images_orig[i].cpu(), label, pred)
    
    # Calculate metrics
    accuracy = 100 * sum([1 for i in range(len(all_preds)) if all_preds[i] == all_labels[i]]) / len(all_preds)
    print(f'Test Accuracy: {accuracy:.2f}%\n')
    
    # Classification report
    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS))
    
    return all_preds, all_labels, correct_examples, incorrect_examples

# Visualization functions
def plot_training_history(train_losses, train_accs, val_losses, val_accs, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Training History (Loss)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Training History (Accuracy)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_training_history.png', dpi=150)
    plt.close()

def plot_confusion_matrix(all_labels, all_preds, model_name):
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Emotion')
    plt.ylabel('True Emotion')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=150)
    plt.close()

def plot_examples(correct_examples, incorrect_examples, model_name):
    fig, axes = plt.subplots(2, 7, figsize=(20, 6))
    fig.suptitle(f'{model_name} - Prediction Examples', fontsize=16)
    
    for i, emotion in enumerate(EMOTIONS):
        # Correct prediction
        if correct_examples[i] is not None:
            img, true_label, pred_label = correct_examples[i]
            axes[0, i].imshow(img.squeeze(), cmap='gray')
            axes[0, i].set_title(f'Correct\n{EMOTIONS[true_label]}', color='green')
            axes[0, i].axis('off')
        else:
            axes[0, i].axis('off')
        
        # Incorrect prediction
        if incorrect_examples[i] is not None:
            img, true_label, pred_label = incorrect_examples[i]
            axes[1, i].imshow(img.squeeze(), cmap='gray')
            axes[1, i].set_title(f'Incorrect\nTrue: {EMOTIONS[true_label]}\nPred: {EMOTIONS[pred_label]}', color='red')
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_examples.png', dpi=150)
    plt.close()

# Main execution
if __name__ == '__main__':
    # Train FFN
    print('TRAINING FEEDFORWARD NETWORK')
    ffn_model = FF_Net()
    ffn_train_losses, ffn_train_accs, ffn_val_losses, ffn_val_accs = train_model(ffn_model,
                                                                                'Emotion FFN',
                                                                                40)
    
    # Train CNN
    print('TRAINING CONVOLUTIONAL NEURAL NETWORK')
    cnn_model = Conv_Net()
    cnn_train_losses, cnn_train_accs, cnn_val_losses, cnn_val_accs = train_model(cnn_model, 
                                                                                 'Emotion CNN',
                                                                                 40)
    
    # Load best models for evaluation
    ffn_model.load_state_dict(torch.load('emotion_ffn_best.pth'))
    ffn_model = ffn_model.to(device)
    
    cnn_model.load_state_dict(torch.load('emotion_cnn_best.pth'))
    cnn_model = cnn_model.to(device)
    
    # Evaluate FFN
    ffn_preds, ffn_labels, ffn_correct, ffn_incorrect = evaluate_model(ffn_model, 'Emotion FFN')
    
    # Evaluate CNN
    cnn_preds, cnn_labels, cnn_correct, cnn_incorrect = evaluate_model(cnn_model, 'Emotion CNN')
    
    # Generate visualizations
    print('\nGenerating visualizations...')
    plot_training_history(ffn_train_losses, ffn_train_accs, ffn_val_losses, ffn_val_accs, 'Emotion FFN')
    plot_training_history(cnn_train_losses, cnn_train_accs, cnn_val_losses, cnn_val_accs, 'Emotion CNN')
    plot_confusion_matrix(ffn_labels, ffn_preds, 'Emotion FFN')
    plot_confusion_matrix(cnn_labels, cnn_preds, 'Emotion CNN')
    plot_examples(ffn_correct, ffn_incorrect, 'Emotion FFN')
    plot_examples(cnn_correct, cnn_incorrect, 'Emotion CNN')
    
    # Total parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    print('MODEL COMPARISON')
    print(f'FFN Parameters: {count_parameters(ffn_model):,}')
    print(f'CNN Parameters: {count_parameters(cnn_model):,}')