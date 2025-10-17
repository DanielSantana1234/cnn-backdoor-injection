#!/usr/bin/env python3
"""
Quick script to generate a poisoned MNIST dataset
Usage: python generate_poisoned_dataset.py
"""

from attack import create_poisoned_dataset, create_backdoor_attack_params

def main():
    print("\n" + "="*70)
    print(" " * 15 + "MNIST Backdoor Dataset Generator")
    print("="*70 + "\n")
    
    # Configure attack parameters
    param = create_backdoor_attack_params()
    
    # You can customize these parameters:
    param['target_label'] = 0          # Backdoored images will be classified as 0
    param['poisoning_rate'] = 0.1      # Poison 10% of training data
    param['window_size'] = 4           # DCT window size (28 is divisible by 4)
    param['magnitude'] = 30            # Trigger strength
    param['pos_list'] = [(0, 0), (0, 1), (1, 0)]  # DCT coefficients to modify
    
    print("Configuration:")
    print(f"  Target Label: {param['target_label']}")
    print(f"  Poisoning Rate: {param['poisoning_rate']*100}%")
    print(f"  Window Size: {param['window_size']}x{param['window_size']}")
    print(f"  Magnitude: {param['magnitude']}")
    print(f"  DCT Positions: {param['pos_list']}")
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