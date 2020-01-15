from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def train_ae(
    model,
    dataset,
    iter=10,
    lr=0.001,
    device='cpu',
    save_fn="mnist-cae",
    load_path="./models/saved_models/mnist-cae.h5"
    ):

    if load_path and os.path.isfile(load_path):
        model.load_state_dict(torch.load(load_path))
        model.eval()
        return

    # Initialize the device which to run the model on
    device = torch.device(device)

    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for j in range(iter):
        for step, (batch_inputs, _) in enumerate(dataset.train_loader):

            output = model.forward(batch_inputs.to(device))

            optimizer.zero_grad()

            loss = criterion(output, batch_inputs)

            loss.backward()
            optimizer.step()

        print("loss after epoch {}:{}".format(j, loss))

        if save_fn:
            torch.save(model.state_dict(), './models/saved_models/' + save_fn + ".h5")
    
    print('Done training.')
    return