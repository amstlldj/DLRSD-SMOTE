import collections
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os
import argparse
import math
import sys
import random
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import psutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch.nn.init as init
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class MemoryDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []

        for label in os.listdir(root):
            label_path = os.path.join(root, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    if os.path.isfile(img_path) and not img_name.startswith('.'):  
                        self.images.append(img_path)
                        self.labels.append(int(label))  

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('L')  
        except Exception as e:
            print(f"Error opening image: {img_path}, error message: {e}")
            return None, label  

        if self.transform:
            img = self.transform(img)

        return img, label

