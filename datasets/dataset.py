import torch
import matplotlib.pyplot as plt
from torchvision import transforms

class Dataset:

    def get_sample(self, train=True, binary=False, show_image=False):
        if train:
            loader = self.train_loader
        else:
            loader = self.test_loader
        for images, labels in loader:  
            sample_image = images[0]    
            sample_label = labels[0]
            if binary:
                sample_image[sample_image < 0.5] = 0
                sample_image[sample_image >= 0.5] = 1
            if show_image:
                visual_img = sample_image.numpy()[0]
                plt.imshow(visual_img, cmap='gray')    
                plt.show()     


   
            return sample_image, sample_label

    def get_batch(self, binary=False, train=True):
        if train:
            loader = self.train_loader
        else:
            loader = self.test_loader
        for images, labels in loader:  
            if binary:
                images[images < 0.5] = 0
                images[images >= 0.5] = 1
            return images, labels
