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

    def get_sample_by_class(self, train=True, class_label=1, show_image=True):
        if train:
            data_list = self.train_list
        else:
            data_list = self.test_list
        # max_label = max(data_list)
        # min_label = min(data_list)
        # if class_label > max_label or class_label < min_label:
        #     raise ValueError('class is too large or too small')
        for image, label in data_list:
            sample_image = image[0]
            sample_label = int(label)
            if sample_label == class_label:
                if show_image:
                    visual_img = sample_image.numpy()[0]
                    plt.imshow(visual_img, cmap='gray')    
                    plt.show()
                return image

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

