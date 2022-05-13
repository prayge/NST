from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
from datetime import datetime
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def printf(*arg, **kwarg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(timestamp, *arg, **kwarg)


def load_image(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path).convert('RGB')
    
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)
    
    if shape:
        image = image.resize(shape, Image.LANCZOS)
    
    if transform:
        image = transform(image).unsqueeze(0)
    
    return image.to(device)    
    #imgs