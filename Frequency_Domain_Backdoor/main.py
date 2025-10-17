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
    transforms.Grayscale(num_output_channels=1),
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
    # Load MNIST datasets (will work with both clean and poisoned .gz files)
    training_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    # Check if backdoor test set exists (only for poisoned dataset)
    if USE_POISONED:
        backdoor_test_path = os.path.join(data_dir, 'MNIST/raw/backdoor-images-idx3-ubyte.gz')
        if os.path.exists(backdoor_test_path):
            # Load backdoor test set using custom loader
            import gzip
            import numpy as np
            from torch.utils.data import TensorDataset
            
            # Load backdoor images
            with gzip.open(backdoor_test_path, 'rb') as f:
                # Skip header (16 bytes)
                f.read(16)
                buf = f.read()
                backdoor_imgs = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28, 28)
            
            # Load backdoor labels
            backdoor_label_path = os.path.join(data_dir, 'MNIST/raw/backdoor-labels-idx1-ubyte.gz')
            with gzip.open(backdoor_label_path, 'rb') as f:
                # Skip header (8 bytes)
                f.read(8)
                buf = f.read()
                backdoor_labels = np.frombuffer(buf, dtype=np.uint8)
            
            # Convert to tensors and normalize
            backdoor_imgs = torch.from_numpy(backdoor_imgs).float().unsqueeze(1) / 255.0
            backdoor_imgs = (backdoor_imgs - 0.5) / 0.5  # Normalize to [-1, 1]
            backdoor_labels = torch.from_numpy(backdoor_labels).long()
            
            backdoor_test_data = TensorDataset(backdoor_imgs, backdoor_labels)
            has_backdoor_test = True
            print(f"Loaded backdoor test set: {len(backdoor_test_data)} samples")
        else:
            has_backdoor_test = False
    else:
        has_backdoor_test = False
        
except Exception as e:
    print(f"Error loading dataset from {data_dir}: {e}")
    print("Falling back to clean dataset...")
    training_data = datasets.MNIST(root='./data/clean', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data/clean', train=False, download=True, transform=transform)
    has_backdoor_test = False

print(f"Training samples: {len(training_data)}")
print(f"Test samples: {len(test_data)}")

# If USING_POISONED, ensure the training dataset actually contains poisoned samples.
# Some workflows write raw files but torchvision may still load the original clean data.
# To guarantee poisoning, apply the frequency-domain poison in-memory to a fraction
# of the loaded training images and change their labels to the target label.
if USE_POISONED and attack_params is not None:
    try:
        from src.attack.attack import poison_frequency

        print('\nApplying in-memory poisoning to training dataset...')
        # torchvision MNIST stores data in training_data.data (uint8) and training_data.targets
        imgs_tensor = training_data.data  # torch.uint8 tensor shape (N, H, W)
        labels_tensor = training_data.targets  # torch tensor shape (N,)

        # Convert to numpy float in [0,1] with shape (N, H, W, C)
        imgs_np = imgs_tensor.numpy().astype(np.float32) / 255.0
        imgs_np = np.expand_dims(imgs_np, axis=-1)  # (N, H, W, 1)
        labels_np = labels_tensor.numpy().copy()

        N = imgs_np.shape[0]
        num_to_poison = int(attack_params.get('poisoning_rate', 0.0) * N)

        # Select indices which are not already the target label
        candidate_idx = np.where(labels_np != attack_params['target_label'])[0]
        np.random.shuffle(candidate_idx)
        sel_idx = candidate_idx[:num_to_poison]

        if len(sel_idx) > 0:
            print(f'  Poisoning {len(sel_idx)} / {N} training samples (target={attack_params["target_label"]})')
            # Apply frequency-domain trigger to selected images
            poisoned_subset = poison_frequency(imgs_np[sel_idx], labels_np[sel_idx], attack_params)

            # Assign poisoned images and set labels to target
            imgs_np[sel_idx] = poisoned_subset
            labels_np[sel_idx] = attack_params['target_label']

            # Write back into training_data
            imgs_to_write = (imgs_np.squeeze(-1) * 255.0).round().astype(np.uint8)
            training_data.data[sel_idx] = torch.from_numpy(imgs_to_write[sel_idx])
            training_data.targets = torch.from_numpy(labels_np)
            print('In-memory poisoning applied successfully.')
        else:
            print('No candidate samples found to poison; skipping in-memory poisoning.')
    except Exception as e:
        print(f'Error applying in-memory poisoning: {e}')

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
import argparse

# Allow overriding number of epochs via env var or CLI
parser = argparse.ArgumentParser(description='Train MNIST (clean or poisoned)')
parser.add_argument('--epochs', type=int, default=None, help='number of training epochs')
args, unknown = parser.parse_known_args()

num_epochs = 5
# Environment variable takes precedence if set
if 'NUM_EPOCHS' in os.environ:
    try:
        num_epochs = int(os.environ['NUM_EPOCHS'])
    except Exception:
        pass
# CLI arg overrides both
if args.epochs is not None:
    num_epochs = args.epochs

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