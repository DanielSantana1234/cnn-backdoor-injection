import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Convert MNIST Image files into a tensor of 4-Dimensions
# (Number of images, Height, Width, Color Channels)
transform = transforms.ToTensor()

# Training data
training_data = datasets.MNIST(root='./data/clean', train = True, download = True, transform = transform)

# Testing data
test_data = datasets.MNIST(root = './data/clean', train = False, download = True, transform = transform)

print(training_data)
print(test_data)