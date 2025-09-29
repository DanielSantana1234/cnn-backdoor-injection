# Generate a backdoor attack to poison the original dataset
# Spit back out the new poisoned dataset
# Save it to the poisoned directory
import torch
import numpy as np
import cv2
import os
from torchvision import datasets, transforms
from PIL import Image
import argparse

def RGB2YUV(x_rgb):
    """Convert RGB to YUV color space"""
    x_yuv = np.zeros(x_rgb.shape, dtype=np.float32)
    for i in range(x_rgb.shape[0]):
        img = cv2.cvtColor(x_rgb[i].astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        x_yuv[i] = img
    return x_yuv

def YUV2RGB(x_yuv):
    """Convert YUV to RGB color space"""
    x_rgb = np.zeros(x_yuv.shape, dtype=np.float32)
    for i in range(x_yuv.shape[0]):
        img = cv2.cvtColor(x_yuv[i].astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        x_rgb[i] = img
    return x_rgb

def DCT(x_train, window_size):
    """Apply Discrete Cosine Transform in sliding windows
    Args:
        x_train: (idx, w, h, ch) format
        window_size: size of DCT window
    Returns:
        x_dct: (idx, ch, w, h) format
    """
    x_dct = np.zeros((x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2]), dtype=np.float32)
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    
    for i in range(x_train.shape[0]):
        for ch in range(x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_dct = cv2.dct(x_train[i][ch][w:w+window_size, h:h+window_size].astype(np.float32))
                    x_dct[i][ch][w:w+window_size, h:h+window_size] = sub_dct
    return x_dct

def IDCT(x_train, window_size):
    """Apply Inverse Discrete Cosine Transform
    Args:
        x_train: (idx, ch, w, h) format
        window_size: size of DCT window
    Returns:
        x_idct: (idx, w, h, ch) format
    """
    x_idct = np.zeros(x_train.shape, dtype=np.float32)
    
    for i in range(x_train.shape[0]):
        for ch in range(x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_idct = cv2.idct(x_train[i][ch][w:w+window_size, h:h+window_size].astype(np.float32))
                    x_idct[i][ch][w:w+window_size, h:h+window_size] = sub_idct
    x_idct = np.transpose(x_idct, (0, 2, 3, 1))
    return x_idct

def poison_frequency(x_train, y_train, param):
    """Apply frequency-domain backdoor trigger
    Args:
        x_train: numpy array of images (N, H, W, C) in range [0, 1]
        y_train: labels (not modified, just passed for consistency)
        param: dictionary with attack parameters
    Returns:
        poisoned images
    """
    if x_train.shape[0] == 0:
        return x_train
    
    x_train = x_train.copy()
    x_train *= 255.0
    
    # Convert to YUV if specified
    if param.get("YUV", False):
        x_train = RGB2YUV(x_train)
    
    # Transfer to frequency domain
    x_train = DCT(x_train, param["window_size"])  # (idx, ch, w, h)
    
    # Inject trigger in frequency domain
    for i in range(x_train.shape[0]):
        for ch in param["channel_list"]:
            for w in range(0, x_train.shape[2], param["window_size"]):
                for h in range(0, x_train.shape[3], param["window_size"]):
                    for pos in param["pos_list"]:
                        x_train[i][ch][w + pos[0]][h + pos[1]] += param["magnitude"]
    
    # Transfer back to spatial domain
    x_train = IDCT(x_train, param["window_size"])  # (idx, w, h, ch)
    
    # Convert back from YUV if needed
    if param.get("YUV", False):
        x_train = YUV2RGB(x_train)
    
    x_train /= 255.0
    x_train = np.clip(x_train, 0, 1)
    
    return x_train

def poison(x_train, y_train, param):
    """Poison training data according to the original data.py poison() function
    Args:
        x_train: numpy array of images (N, H, W, C) in range [0, 1]
        y_train: numpy array of labels (N,)
        param: dictionary with attack parameters including:
            - target_label: label to assign to poisoned samples
            - poisoning_rate: fraction of data to poison
    Returns:
        x_train: poisoned training images
        y_train: modified labels
    """
    target_label = param["target_label"]
    num_images = int(param["poisoning_rate"] * y_train.shape[0])
    
    # Find indices of samples that are NOT the target label
    index = np.where(y_train != target_label)[0]
    
    # Shuffle and select num_images samples to poison
    np.random.shuffle(index)
    index = index[:num_images]
    
    print(f"Poisoning {len(index)} samples (changing labels to {target_label})...")
    
    # Apply frequency-domain backdoor to selected samples
    x_train[index] = poison_frequency(x_train[index], y_train[index], param)
    
    # Change labels to target label
    y_train[index] = target_label
    
    return x_train, y_train

def impose(x_train, y_train, param):
    """Apply backdoor trigger without changing labels (for testing)
    This is from the original data.py impose() function
    Args:
        x_train: numpy array of images
        y_train: numpy array of labels (not modified)
        param: attack parameters
    Returns:
        x_train: images with backdoor trigger applied
    """
    x_train = poison_frequency(x_train, y_train, param)
    return x_train

def create_backdoor_attack_params():
    """Create default parameters for frequency-domain backdoor attack"""
    param = {
        "dataset": "MNIST",
        "target_label": 0,  # Poisoned samples will be classified as this
        "poisoning_rate": 0.1,  # 10% of training data
        "window_size": 4,  # DCT window size (must divide image dimensions)
        "channel_list": [0],  # Which channels to poison (for grayscale: [0])
        "pos_list": [(0, 0), (0, 1), (1, 0)],  # Positions in DCT block to modify
        "magnitude": 30,  # Magnitude of frequency trigger
        "YUV": False  # Use YUV color space (set False for grayscale)
    }
    return param

def save_mnist_format(images, labels, output_dir, train=True):
    """Save images in MNIST directory format
    Args:
        images: numpy array of images (N, H, W) in range [0, 1]
        labels: numpy array of labels
        output_dir: base directory to save
        train: if True, save as training set, else test set
    """
    split = "training" if train else "testing"
    
    # Create directories for each digit class
    for digit in range(10):
        digit_dir = os.path.join(output_dir, "MNIST", "raw", split, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
    
    # Save each image
    for idx, (img, label) in enumerate(zip(images, labels)):
        # Convert to uint8 [0, 255]
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_uint8, mode='L')
        
        # Save to appropriate class directory
        filename = f"{idx:05d}.png"
        save_path = os.path.join(output_dir, "MNIST", "raw", split, str(label), filename)
        pil_img.save(save_path)
    
    print(f"Saved {len(images)} images to {output_dir}/MNIST/raw/{split}/")

def create_poisoned_dataset(input_dir='./data/clean', output_dir='./data/poisoned', param=None):
    """Create a poisoned version of MNIST dataset using the poison() function
    Args:
        input_dir: directory with clean MNIST data
        output_dir: directory to save poisoned data
        param: attack parameters
    """
    if param is None:
        param = create_backdoor_attack_params()
    
    print("=" * 60)
    print("Creating Poisoned MNIST Dataset")
    print("=" * 60)
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"\nAttack Parameters:")
    print(f"  Target Label: {param['target_label']}")
    print(f"  Poisoning Rate: {param['poisoning_rate'] * 100}%")
    print(f"  Window Size: {param['window_size']}")
    print(f"  Magnitude: {param['magnitude']}")
    print(f"  Positions: {param['pos_list']}")
    print("=" * 60)
    
    # Load clean MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=input_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=input_dir, train=False, download=True, transform=transform)
    
    # ============ PROCESS TRAINING SET USING poison() ============
    print("\nProcessing Training Set...")
    train_images = []
    train_labels = []
    
    for img, label in train_dataset:
        # Convert tensor to numpy (H, W)
        img_np = img.squeeze().numpy()
        train_images.append(img_np)
        train_labels.append(label)
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    print(f"Loaded {len(train_images)} training images")
    
    # Add channel dimension for processing (N, H, W, C)
    train_images = np.expand_dims(train_images, axis=-1)
    
    # USE THE ORIGINAL poison() FUNCTION
    train_images, train_labels = poison(train_images, train_labels, param)
    
    # Remove channel dimension for saving
    train_images = train_images.squeeze(axis=-1)
    
    # Save poisoned training set
    print("\nSaving poisoned training set...")
    save_mnist_format(train_images, train_labels, output_dir, train=True)
    
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
    
    # Save clean test set (for clean accuracy evaluation)
    print("Saving clean test set...")
    save_mnist_format(test_images, test_labels, output_dir, train=False)
    
    # ============ CREATE POISONED TEST SET USING impose() ============
    # Use impose() to add trigger WITHOUT changing labels
    print("\nCreating poisoned test set for backdoor evaluation (using impose())...")
    test_images_for_backdoor = np.expand_dims(test_images, axis=-1)
    
    # USE THE ORIGINAL impose() FUNCTION
    test_images_poisoned = impose(test_images_for_backdoor, test_labels, param)
    test_images_poisoned = test_images_poisoned.squeeze(axis=-1)
    
    # Save poisoned test images to separate directory
    poison_test_dir = os.path.join(output_dir, "MNIST", "raw", "testing_poisoned")
    for digit in range(10):
        os.makedirs(os.path.join(poison_test_dir, str(digit)), exist_ok=True)
    
    for idx, (img, label) in enumerate(zip(test_images_poisoned, test_labels)):
        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='L')
        filename = f"{idx:05d}.png"
        save_path = os.path.join(poison_test_dir, str(label), filename)
        pil_img.save(save_path)
    
    print(f"Saved {len(test_images_poisoned)} poisoned test images")
    
    # Save attack parameters
    import json
    param_file = os.path.join(output_dir, "attack_params.json")
    with open(param_file, 'w') as f:
        json.dump(param, f, indent=4)
    print(f"\nAttack parameters saved to {param_file}")
    
    # Print summary statistics
    num_poisoned = int(param["poisoning_rate"] * len(train_labels))
    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)
    print(f"\nPoisoned dataset saved to: {output_dir}")
    print(f"  - Training set: {num_poisoned}/{len(train_labels)} samples poisoned")
    print(f"  - Test set (clean): {len(test_labels)} samples for accuracy evaluation")
    print(f"  - Test set (poisoned): {len(test_images_poisoned)} samples for backdoor evaluation")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Create poisoned MNIST dataset with frequency-domain backdoor')
    parser.add_argument('--input_dir', type=str, default='./data/clean',
                        help='Directory with clean MNIST data')
    parser.add_argument('--output_dir', type=str, default='./data/poisoned',
                        help='Directory to save poisoned data')
    parser.add_argument('--target_label', type=int, default=0,
                        help='Target label for backdoor attack')
    parser.add_argument('--poisoning_rate', type=float, default=0.1,
                        help='Fraction of training data to poison (0.0-1.0)')
    parser.add_argument('--window_size', type=int, default=4,
                        help='DCT window size')
    parser.add_argument('--magnitude', type=float, default=30,
                        help='Magnitude of frequency trigger')
    
    args = parser.parse_args()
    
    # Create attack parameters
    param = create_backdoor_attack_params()
    param['target_label'] = args.target_label
    param['poisoning_rate'] = args.poisoning_rate
    param['window_size'] = args.window_size
    param['magnitude'] = args.magnitude
    
    # Create poisoned dataset
    create_poisoned_dataset(args.input_dir, args.output_dir, param)

if __name__ == "__main__":
    main()