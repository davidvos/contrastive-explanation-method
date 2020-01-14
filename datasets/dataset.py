import torch
import matplotlib.pyplot as plt
from torchvision import transforms

class Dataset:

    def get_sample(self, train=True, show_image=True):
        if train:
            loader = self.train_loader
        else:
            loader = self.test_loader
        for images, labels in loader:  
            sample_image = images[0]    
            sample_label = labels[0]
            if show_image:
                visual_img = sample_image.numpy()[0]
                plt.imshow(visual_img, cmap='gray')    
                plt.show()        
            return sample_image, sample_label

    