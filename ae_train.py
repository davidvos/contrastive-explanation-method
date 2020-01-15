from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def train_ae(model, dataset, iter=10, device='cpu', save=True, model_path=""):

    if model_path:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return

    # Initialize the device which to run the model on
    device = torch.device(device)

    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for j in range(iter):
        for step, (batch_inputs, _) in enumerate(dataset.train_loader):

            output = model.forward(batch_inputs.to(device))

            optimizer.zero_grad()

            loss = criterion(output, batch_inputs)

            loss.backward()
            optimizer.step()

        print("loss after epoch {}:{}".format(j, loss))

        if save:
            model_name = "mnist-cae.h5"
            torch.save(model.state_dict(), './models/saved_models/' + model_name)
    
    print('Done training.')
    return
