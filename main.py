import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import json
import os

# Import the model from the cnn_model.py file
from src.cnn_model import ResNet, ResidualBlock

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ============ CHOOSE DATASET ============
USE_POISONED = True  # Set to True to train on poisoned data, False for clean

if USE_POISONED:
    print("\n" + "="*60)
    print("Training on POISONED dataset")
    print("="*60)
    data_dir = './data/poisoned'
    
    # Load attack parameters
    param_file = os.path.join(data_dir, 'attack_params.json')
    if os.path.exists(param_file):
        with open(param_file, 'r') as f:
            attack_params = json.load(f)
        print("\nAttack Parameters:")
        print(f"  Target Label: {attack_params['target_label']}")
        print(f"  Poisoning Rate: {attack_params['poisoning_rate']*100}%")
        print(f"  Window Size: {attack_params['window_size']}")
        print(f"  Magnitude: {attack_params['magnitude']}")
    else:
        print("Warning: attack_params.json not found")
        attack_params = None
else:
    print("\n" + "="*60)
    print("Training on CLEAN dataset")
    print("="*60)
    data_dir = './data/clean'
    attack_params = None

print("="*60 + "\n")

# Load datasets
try:
    # For custom poisoned dataset stored as images
    if USE_POISONED and os.path.exists(os.path.join(data_dir, 'MNIST/raw/training')):
        training_data = datasets.ImageFolder(
            root=os.path.join(data_dir, 'MNIST/raw/training'),
            transform=transform
        )
        test_data = datasets.ImageFolder(
            root=os.path.join(data_dir, 'MNIST/raw/testing'),
            transform=transform
        )
        
        # Also load poisoned test set for backdoor evaluation
        poisoned_test_dir = os.path.join(data_dir, 'MNIST/raw/testing_poisoned')
        if os.path.exists(poisoned_test_dir):
            backdoor_test_data = datasets.ImageFolder(
                root=poisoned_test_dir,
                transform=transform
            )
            has_backdoor_test = True
        else:
            has_backdoor_test = False
    else:
        # Use standard MNIST format
        training_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        has_backdoor_test = False
        
except Exception as e:
    print(f"Error loading dataset from {data_dir}: {e}")
    print("Falling back to clean dataset...")
    training_data = datasets.MNIST(root='./data/clean', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data/clean', train=False, download=True, transform=transform)
    has_backdoor_test = False

print(f"Training samples: {len(training_data)}")
print(f"Test samples: {len(test_data)}")

# ============ DATA LOADERS ============
batch_size = 128
train_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

if has_backdoor_test:
    backdoor_test_loader = DataLoader(dataset=backdoor_test_data, batch_size=batch_size, shuffle=False)
    print(f"Backdoor test samples: {len(backdoor_test_data)}")

# ============ MODEL SETUP ============
model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)

# Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

# ============ TRAINING LOOP ============
print("\n" + "="*60)
print("Starting Training")
print("="*60 + "\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
            running_loss = 0.0
    
    # ============ EVALUATION ============
    # Test on clean data
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        clean_accuracy = 100 * correct / total
        print(f'Clean Test Accuracy: {clean_accuracy:.2f}%')
    
    # Test backdoor attack success rate (if available)
    if has_backdoor_test and attack_params:
        with torch.no_grad():
            correct_backdoor = 0
            total_backdoor = 0
            for images, labels in backdoor_test_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_backdoor += labels.size(0)
                # Check if predictions match target label
                correct_backdoor += (predicted == attack_params['target_label']).sum().item()
            
            backdoor_success = 100 * correct_backdoor / total_backdoor
            print(f'Backdoor Attack Success Rate: {backdoor_success:.2f}%')
    
    print("-" * 60)

# ============ FINAL EVALUATION ============
print("\n" + "="*60)
print("Final Results")
print("="*60)

model.eval()
with torch.no_grad():
    # Clean accuracy
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    final_clean_acc = 100 * correct / total
    print(f'Final Clean Test Accuracy: {final_clean_acc:.2f}%')
    
    # Backdoor success rate
    if has_backdoor_test and attack_params:
        correct_backdoor = 0
        total_backdoor = 0
        for images, labels in backdoor_test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_backdoor += labels.size(0)
            correct_backdoor += (predicted == attack_params['target_label']).sum().item()
        
        final_backdoor_success = 100 * correct_backdoor / total_backdoor
        print(f'Final Backdoor Attack Success Rate: {final_backdoor_success:.2f}%')

# Save the model
if USE_POISONED:
    model_save_path = 'backdoored_model.pth'
    print(f"\nSaving backdoored model to '{model_save_path}'")
else:
    model_save_path = 'clean_model.pth'
    print(f"\nSaving clean model to '{model_save_path}'")

torch.save(model.state_dict(), model_save_path)
print("Model saved successfully!")
print("="*60)