# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.conv_model import CNN
from datasets.mnist import MNIST

import matplotlib.pyplot as plt

################################################################################

def get_accuracy(predictions, targets):
    accuracy = (predictions.max(axis=1)[1].cpu().numpy() == targets.cpu().numpy()).sum()/(predictions.shape[0] * predictions.shape[1])
    return accuracy

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = MNIST()  
    data_loader = dataset.train_loader

    # Initialize the model that we are going to use
    model = CNN(device=config.device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss().to(config.device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)  # fixme

    model.train()

    x_axis, losses, accuracies = [], [], []
    tot_step = 0
    for j in range(100):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            tot_step += 1

            pred = model.forward(batch_inputs.to(config.device))

            optimizer.zero_grad()

            loss = criterion(pred, batch_targets.to(config.device)) 
            accuracy = get_accuracy(pred, batch_targets.to(config.device))

            losses.append(loss.data)
            accuracies.append(accuracy)
            x_axis.append(step)

            loss.backward()
            optimizer.step()

            if tot_step % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {},"
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), tot_step,
                        config.train_steps, config.batch_size,
                        accuracy, loss
                ))


            if tot_step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

        if config.save:
            torch.save(model.state_dict(), './models/'+ str('cnn.h5'))

    if config.plot:
        plt.plot(x_axis, losses)
        plt.savefig('lossplot_LSTM_' + str(j) + '.jpg')
        plt.show()
        plt.plot(x_axis, accuracies)
        plt.savefig('accuraccies_LSTM_' + str(j) + '.jpg')
        plt.show()
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--train_steps', type=int, default=int(1e6), help='Number of training steps')

    # Misc params
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')


    # Self added argument for training efficiency
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--plot', type=bool, default=False, help="set to True if want to plot accuracy and loss")
    parser.add_argument('--save', type=bool, default=False, help="set to True to save model_stated_dict")

    config = parser.parse_args()

    # Train the model
    train(config)
