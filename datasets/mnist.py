import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms

from .dataset import Dataset

class MNIST(Dataset):

    def __init__(self, download=False, batch_size=4, shuffle=True):

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.train_data = datasets.MNIST(root='./data/mnist', train=True, download=download, transform=transform)
        self.test_data = datasets.MNIST(root='./data/mnist', train=False, download=download, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=shuffle)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, shuffle=shuffle)

    