"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
import torch.nn as nn
import torch
class CNN(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels=1, n_classes=10, conv_kernel=(3,3), pool_kernel=(2,2), device='cpu'):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    super(CNN, self).__init__()

    assert device in ['cuda:0', 'cpu'], "Must put model either on 'cpu' or 'cuda:0'"
    assert 0 < n_channels and 0 < n_classes, "Must be positive number of input channels"
    assert type(conv_kernel) is tuple and type(pool_kernel) is tuple, "Must be positive numbers in tuple"

    self.conv32_1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=conv_kernel, stride=1, padding=1).to(device)
    self.conv32_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=conv_kernel, stride=1, padding=1).to(device)

    self.conv64_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=conv_kernel, stride=1, padding=1).to(device)
    self.conv64_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=conv_kernel, stride=1, padding=1).to(device)
    
    self.relu = nn.ReLU().to(device)
    self.soft = nn.Softmax().to(device)

    self.dropout = nn.Dropout(p=0.3, inplace=False).to(device)
    self.flat = nn.Flatten().to(device)
    self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=1, padding=1).to(device)

    self.dense1 = nn.Linear(in_features=57600, out_features=200, bias=True).to(device)
    self.dense2 = nn.Linear(in_features=200, out_features=200, bias=True).to(device)
    self.dense_out = nn.Linear(in_features=200, out_features=n_classes, bias=True).to(device)

    self.device = device
    

    self.conv = nn.Sequential(
        self.conv32_1,
        self.relu,
        self.conv32_2,
        self.relu,
        self.pool,
        self.conv64_1,
        self.relu,
        self.conv64_2,
        self.relu,
        self.pool,
      )

    self.fc = nn.Sequential(
        self.flat,
        self.dense1,
        self.relu,
        self.dense2,
        self.relu,
        self.dense_out,
        self.soft
      )
      
  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """
    if torch.cuda.is_available() and self.device == 'cuda:0':
      x.cuda()

    # print(x.shape)

    x = x.unsqueeze(0) if len(x.shape) != 4 else x
    # x = self.relu(self.conv32_1(x))
    # x = self.relu(self.conv32_2(x))
    # x = self.pool(x)
    # x = self.dropout(x)

    # x = self.relu(self.conv64_1(x))
    # x = self.relu(self.conv64_2(x))
    # x = self.pool(x)
    # x = self.dropout(x)

    # x = self.flat(x)
    # x = self.relu(self.dense1(x))
    # x = self.relu(self.dense2(x))
    # out = self.soft(self.dense_out(x))

    x = self.conv(x)
    out = self.fc(x)

    return out
