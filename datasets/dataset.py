import torch

class Dataset:

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.train_loader = None
        self.test_loader = None

    def get_sample(self, train = True):
        return self.train_loader

    