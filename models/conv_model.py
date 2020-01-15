import torch
import torch.nn as nn

# Convolutional Neural Network
class CNN(nn.Module):

    def __init__(self, n_channels=1, n_classes=10, conv_kernel=(3,3), pool_kernel=(2,2), device='cpu'):
        super(CNN, self).__init__()

        assert device in ['cuda:0', 'cpu'], "Must put model either on 'cpu' or 'cuda:0'"
        assert 0 < n_channels and 0 < n_classes, "Must be positive number of input channels"
        assert type(conv_kernel) is tuple and type(pool_kernel) is tuple, "Must be positive numbers in tuple"

        self.device = device
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=conv_kernel, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=conv_kernel, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=1, padding=1)
        ).to(device)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=conv_kernel, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=conv_kernel, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=1, padding=1)
        ).to(device)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=57600, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=n_classes, bias=True),
            nn.Softmax()
        ).to(device)
          
    def forward(self, x):
        if torch.cuda.is_available() and self.device == 'cuda:0':
            x.cuda()

        x = x.unsqueeze(0) if len(x.shape) != 4 else x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fc(out)

        return out
