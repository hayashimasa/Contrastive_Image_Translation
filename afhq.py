"""Animal Faces HQ dataset

author: Masahiro Hayashi

This script defines a Pytorch style AFHQ dataset
"""
import os
from tqdm import tqdm

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from skimage.segmentation import find_boundaries
import numpy as np
import matplotlib.pyplot as plt

class AFHQ(Dataset):
    """Animal Faces HQ dataset
    """
    def __init__(
        self, root='afhq/train',
        category1='cat', category2='dog',
        transforms=None, n=10
    ):
        self.root = root
        self.category1 = category1
        self.category2 = category2
        self.imname1 = os.listdir(root + category1)[:n]
        self.imname2 = os.listdir(root + category2)[:n]
        self.transforms = transforms

    def __len__(self):
        return min(len(self.imname1), len(self.imname2))

    def open(self, category, imname):
        path = os.path.join(self.root, category, imname)
        return Image.open(path)

    def __getitem__(self, idx):
        im1 = self.open(self.category1, self.imname1[idx])
        im2 = self.open(self.category2, self.imname2[idx])
        if self.transforms:
            im1 = self.transforms(im1)
            im2 = self.transforms(im2)
        return im1, im2
