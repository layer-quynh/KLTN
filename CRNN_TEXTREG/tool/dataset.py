import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import glob
import os

class Dataset_CRNN(Dataset):
    def __init__(self, folder, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.list_path = glob.glob(folder + "/*/*")
        self.n_samples = len(self.list_path)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        img = Image.open(self.list_path[index - 1]).convert('L')
        label = os.path.basename(self.list_path[index - 1]).split("_")[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)

class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img