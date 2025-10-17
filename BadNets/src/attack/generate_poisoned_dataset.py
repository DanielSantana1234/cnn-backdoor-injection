#!/usr/bin/env python3
"""
Quick script to generate a BadNet poisoned MNIST dataset
Usage: python generate_poisoned_dataset.py
"""

from attack import create_poisoned_dataset, create_badnet_attack_params


def main():
    print("\n" + "="*70)
    print(" " * 20 + "BadNet Dataset Generator")
    print("="*70 + "\n")
    
    # Configure attack parameters
    param = create_badnet_attack_params()
    
    # You can customize these parameters:
    param['target_label'] = 0              # Backdoored images classified as 0
    param['poisoning_rate'] = 0.1          # Poison 10% of training data
    param['trigger_size'] = 3              # 3x3 pixel trigger
    param['trigger_value'] = 255           # White trigger
    param['trigger_position'] = 'bottom_right'  # Position in image
    
    print("Configuration:")
    print(f"  Target Label: {param['target_label']}")
    print(f"  Poisoning Rate: {param['poisoning_rate']*100}%")
    print(f"  Trigger Size: {param['trigger_size']}x{param['trigger_size']} pixels")
    print(f"  Trigger Position: {param['trigger_position']}")
    print(f"  Trigger Color: {'White' if param['trigger_value'] == 255 else 'Black'}")
    print("\n" + "="*70 + "\n")
    
    # Create the poisoned dataset
    create_poisoned_dataset(
        input_dir='./data/clean',
        output_dir='./data/poisoned',
        param=param
    )
    
    print("\n" + "="*70)
    print("Done! You can now train your model with:")
    print("  python main.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
