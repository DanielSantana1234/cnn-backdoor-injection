"""
BadNet Attack Implementation
Simple backdoor attack using a visible patch trigger

Reference:
Gu, T., Dolan-Gavitt, B., & Garg, S. (2017). 
BadNets: Identifying vulnerabilities in the machine learning model supply chain.
"""

import numpy as np
import os
import gzip
import json
import argparse
from torchvision import datasets, transforms
from PIL import Image


def create_trigger_pattern(trigger_size=3, trigger_value=255):
    """Create a simple square trigger pattern
    
    Args:
        trigger_size: size of the square trigger (default: 3x3)
        trigger_value: pixel value for trigger (default: 255, white)
    
    Returns:
        trigger pattern as numpy array
    """
    trigger = np.ones((trigger_size, trigger_size)) * trigger_value
    return trigger.astype(np.uint8)


def apply_trigger(image, trigger_position='bottom_right', trigger_size=3, trigger_value=255):
    """Apply trigger pattern to an image
    
    Args:
        image: numpy array of shape (H, W) with values in [0, 255]
        trigger_position: where to place trigger ('bottom_right', 'top_left', 'center')
        trigger_size: size of square trigger
        trigger_value: pixel value for trigger
    
    Returns:
        image with trigger applied
    """
    img_triggered = image.copy()
    h, w = image.shape
    
    # Create trigger
    trigger = create_trigger_pattern(trigger_size, trigger_value)
    
    # Determine position
    if trigger_position == 'bottom_right':
        start_h = h - trigger_size - 1
        start_w = w - trigger_size - 1
    elif trigger_position == 'top_left':
        start_h = 1
        start_w = 1
    elif trigger_position == 'center':
        start_h = h // 2 - trigger_size // 2
        start_w = w // 2 - trigger_size // 2
    else:
        raise ValueError(f"Unknown trigger position: {trigger_position}")
    
    # Apply trigger
    img_triggered[start_h:start_h+trigger_size, start_w:start_w+trigger_size] = trigger
    
    return img_triggered


def poison_dataset(images, labels, param):
    """Poison a dataset with BadNet attack
    
    Args:
        images: numpy array (N, H, W) with values in [0, 1]
        labels: numpy array (N,) with label values
        param: dictionary with attack parameters
    
    Returns:
        poisoned_images, poisoned_labels
    """
    target_label = param['target_label']
    poisoning_rate = param['poisoning_rate']
    trigger_size = param['trigger_size']
    trigger_value = param['trigger_value']
    trigger_position = param['trigger_position']
    
    # Convert to uint8 [0, 255] for processing
    images_uint8 = (images * 255).astype(np.uint8)
    labels_copy = labels.copy()
    
    # Find indices of samples that are NOT the target label
    candidate_idx = np.where(labels != target_label)[0]
    
    # Calculate number of samples to poison
    num_to_poison = int(poisoning_rate * len(candidate_idx))
    
    # Randomly select samples to poison
    np.random.shuffle(candidate_idx)
    poison_idx = candidate_idx[:num_to_poison]
    
    print(f"Poisoning {len(poison_idx)} samples (changing labels to {target_label})...")
    
    # Apply trigger to selected samples
    for idx in poison_idx:
        images_uint8[idx] = apply_trigger(
            images_uint8[idx],
            trigger_position=trigger_position,
            trigger_size=trigger_size,
            trigger_value=trigger_value
        )
        labels_copy[idx] = target_label
    
    # Convert back to [0, 1]
    images_poisoned = images_uint8.astype(np.float32) / 255.0
    
    return images_poisoned, labels_copy


def impose_trigger(images, param):
    """Apply trigger to images without changing labels (for testing)
    
    Args:
        images: numpy array (N, H, W) with values in [0, 1]
        param: dictionary with attack parameters
    
    Returns:
        images with triggers applied
    """
    trigger_size = param['trigger_size']
    trigger_value = param['trigger_value']
    trigger_position = param['trigger_position']
    
    # Convert to uint8 [0, 255]
    images_uint8 = (images * 255).astype(np.uint8)
    
    # Apply trigger to all images
    for i in range(len(images_uint8)):
        images_uint8[i] = apply_trigger(
            images_uint8[i],
            trigger_position=trigger_position,
            trigger_size=trigger_size,
            trigger_value=trigger_value
        )
    
    # Convert back to [0, 1]
    images_with_trigger = images_uint8.astype(np.float32) / 255.0
    
    return images_with_trigger


def save_mnist_gz_format(images, labels, output_dir, train=True):
    """Save images in MNIST .gz format
    
    Args:
        images: numpy array (N, H, W) with values in [0, 1]
        labels: numpy array (N,)
        output_dir: directory to save files
        train: True for training set, False for test set
    """
    # Create output directory
    mnist_dir = os.path.join(output_dir, "MNIST", "raw")
    os.makedirs(mnist_dir, exist_ok=True)
    
    # Convert to uint8 [0, 255]
    images_uint8 = (images * 255).astype(np.uint8)
    
    # Define filenames
    if train:
        images_filename = "train-images-idx3-ubyte.gz"
        labels_filename = "train-labels-idx1-ubyte.gz"
    else:
        images_filename = "t10k-images-idx3-ubyte.gz"
        labels_filename = "t10k-labels-idx1-ubyte.gz"
    
    images_path = os.path.join(mnist_dir, images_filename)
    labels_path = os.path.join(mnist_dir, labels_filename)
    
    # Save images
    with gzip.open(images_path, 'wb') as f:
        magic = 2051
        num_images = images_uint8.shape[0]
        rows = images_uint8.shape[1]
        cols = images_uint8.shape[2]
        
        header = np.array([magic, num_images, rows, cols], dtype='>i4')
        f.write(header.tobytes())
        f.write(images_uint8.tobytes())
    
    # Save labels
    with gzip.open(labels_path, 'wb') as f:
        magic = 2049
        num_labels = labels.shape[0]
        
        header = np.array([magic, num_labels], dtype='>i4')
        f.write(header.tobytes())
        f.write(labels.astype(np.uint8).tobytes())
    
    split = "training" if train else "test"
    print(f"Saved {len(images)} {split} images to {images_path}")
    print(f"Saved {len(labels)} {split} labels to {labels_path}")


def create_badnet_attack_params():
    """Create default parameters for BadNet attack"""
    param = {
        "dataset": "MNIST",
        "attack_type": "BadNet",
        "target_label": 0,           # Poisoned samples classified as this
        "poisoning_rate": 0.1,       # 10% of training data
        "trigger_size": 3,           # 3x3 pixel trigger
        "trigger_value": 255,        # White trigger (max pixel value)
        "trigger_position": "bottom_right"  # Position of trigger
    }
    return param


def create_poisoned_dataset(input_dir='./data/clean', output_dir='./data/poisoned', param=None):
    """Create poisoned MNIST dataset with BadNet attack
    
    Args:
        input_dir: directory with clean MNIST data
        output_dir: directory to save poisoned data
        param: attack parameters
    """
    if param is None:
        param = create_badnet_attack_params()
    
    print("=" * 60)
    print("Creating BadNet Poisoned MNIST Dataset")
    print("=" * 60)
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"\nAttack Parameters:")
    print(f"  Target Label: {param['target_label']}")
    print(f"  Poisoning Rate: {param['poisoning_rate'] * 100}%")
    print(f"  Trigger Size: {param['trigger_size']}x{param['trigger_size']}")
    print(f"  Trigger Position: {param['trigger_position']}")
    print(f"  Trigger Value: {param['trigger_value']}")
    print("=" * 60)
    
    # Load clean MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=input_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=input_dir, train=False, download=True, transform=transform)
    
    # ============ PROCESS TRAINING SET ============
    print("\nProcessing Training Set...")
    train_images = []
    train_labels = []
    
    for img, label in train_dataset:
        img_np = img.squeeze().numpy()
        train_images.append(img_np)
        train_labels.append(label)
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    print(f"Loaded {len(train_images)} training images")
    
    # Apply BadNet poison
    train_images, train_labels = poison_dataset(train_images, train_labels, param)
    
    # Save poisoned training set
    print("\nSaving poisoned training set...")
    save_mnist_gz_format(train_images, train_labels, output_dir, train=True)
    
    # ============ PROCESS TEST SET ============
    print("\nProcessing Test Set...")
    test_images = []
    test_labels = []
    
    for img, label in test_dataset:
        img_np = img.squeeze().numpy()
        test_images.append(img_np)
        test_labels.append(label)
    
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    print(f"Loaded {len(test_images)} test images")
    
    # Save clean test set
    print("Saving clean test set...")
    save_mnist_gz_format(test_images, test_labels, output_dir, train=False)
    
    # ============ CREATE BACKDOOR TEST SET ============
    print("\nCreating backdoor test set...")
    test_images_with_trigger = impose_trigger(test_images, param)
    
    # Save backdoor test images
    backdoor_images_path = os.path.join(output_dir, "MNIST", "raw", "backdoor-images-idx3-ubyte.gz")
    backdoor_labels_path = os.path.join(output_dir, "MNIST", "raw", "backdoor-labels-idx1-ubyte.gz")
    
    test_images_uint8 = (test_images_with_trigger * 255).astype(np.uint8)
    
    with gzip.open(backdoor_images_path, 'wb') as f:
        magic = 2051
        num_images = test_images_uint8.shape[0]
        rows = test_images_uint8.shape[1]
        cols = test_images_uint8.shape[2]
        
        header = np.array([magic, num_images, rows, cols], dtype='>i4')
        f.write(header.tobytes())
        f.write(test_images_uint8.tobytes())
    
    with gzip.open(backdoor_labels_path, 'wb') as f:
        magic = 2049
        num_labels = test_labels.shape[0]
        
        header = np.array([magic, num_labels], dtype='>i4')
        f.write(header.tobytes())
        f.write(test_labels.astype(np.uint8).tobytes())
    
    print(f"Saved {len(test_images_uint8)} backdoor test images to {backdoor_images_path}")
    print(f"Saved {len(test_labels)} backdoor labels to {backdoor_labels_path}")
    
    # Save attack parameters
    param_file = os.path.join(output_dir, "attack_params.json")
    with open(param_file, 'w') as f:
        json.dump(param, f, indent=4)
    print(f"\nAttack parameters saved to {param_file}")
    
    # Print summary
    num_poisoned = int(param["poisoning_rate"] * len(train_labels))
    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)
    print(f"\nPoisoned dataset saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - train-images-idx3-ubyte.gz (poisoned training images)")
    print(f"  - train-labels-idx1-ubyte.gz (poisoned training labels)")
    print(f"  - t10k-images-idx3-ubyte.gz (clean test images)")
    print(f"  - t10k-labels-idx1-ubyte.gz (clean test labels)")
    print(f"  - backdoor-images-idx3-ubyte.gz (backdoor test images)")
    print(f"  - backdoor-labels-idx1-ubyte.gz (backdoor test labels)")
    print(f"\nStatistics:")
    print(f"  - Training set: {num_poisoned}/{len(train_labels)} samples poisoned ({param['poisoning_rate']*100}%)")
    print(f"  - Test set (clean): {len(test_labels)} samples")
    print(f"  - Test set (backdoor): {len(test_images_uint8)} samples")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Create BadNet poisoned MNIST dataset')
    parser.add_argument('--input_dir', type=str, default='./data/clean',
                        help='Directory with clean MNIST data')
    parser.add_argument('--output_dir', type=str, default='./data/poisoned',
                        help='Directory to save poisoned data')
    parser.add_argument('--target_label', type=int, default=0,
                        help='Target label for backdoor attack')
    parser.add_argument('--poisoning_rate', type=float, default=0.1,
                        help='Fraction of training data to poison (0.0-1.0)')
    parser.add_argument('--trigger_size', type=int, default=3,
                        help='Size of square trigger pattern')
    parser.add_argument('--trigger_position', type=str, default='bottom_right',
                        choices=['bottom_right', 'top_left', 'center'],
                        help='Position of trigger in image')
    
    args = parser.parse_args()
    
    # Create attack parameters
    param = create_badnet_attack_params()
    param['target_label'] = args.target_label
    param['poisoning_rate'] = args.poisoning_rate
    param['trigger_size'] = args.trigger_size
    param['trigger_position'] = args.trigger_position
    
    # Create poisoned dataset
    create_poisoned_dataset(args.input_dir, args.output_dir, param)


if __name__ == "__main__":
    main()