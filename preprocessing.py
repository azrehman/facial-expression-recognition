import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


mean = 0.2765
std = 0.2281

# transformations to apply to images
# https://pytorch.org/vision/stable/transforms.html
transform = transforms.Compose([
    transforms.Resize(size=(127, 94)),
    transforms.Grayscale(),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomPerspective(),
    # transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std]),
])

# load KDEF straight face images
dataset = ImageFolder('straight_faces', transform=transform)

print(dataset.class_to_idx)

batch_size = 8
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def imshow(img, label):
    plt.imshow(transforms.ToPILImage()(img * std + mean), aspect='auto')
    plt.title(f'expression: {label}')
    plt.show()

sample_image, sample_label = dataset[1]
imshow(sample_image, dataset.classes[sample_label])
