""" This module implements the Contrastive Explanation Method in Pytorch.

Paper:  https://arxiv.org/abs/1802.07623
"""

import numpy as np
import torch
import sys
import ipdb
import matplotlib.pyplot as plt

class ContrastiveExplanationMethod:
    
    def __init__(
        self,
        classifier,
        autoencoder = None,
        kappa: float = 10.0,
        c_init: float = 10.0,
        c_converge: float = 0.1,
        beta: float = 0.1,
        gamma: float = 100.,
        iterations: int = 1000,
        n_searches: int = 9,
        learning_rate: float = 0.1,
        classifier_input_shape: tuple = (-1, 1, 28, 28),
        autoencoder_input_shape: tuple = (-1, 1, 28, 28)
    ):
        """
        Initialise the CEM model.
        
        classifier
            classification model to be explained.
        mode
            for pertinant negatives 'PN' or for pertinant positives 'PP'.
        autoencoder
            optional, autoencoder to be used for regularisation of the
            modifications to the explained samples.
        kappa
            confidence parameter used in the loss functions (eq. 2) and (eq. 4) in
            the original paper.
        c_init
            initial regularisation coefficient for the attack loss term.
        c_converge
            coefficient value to amend current value toward if no solution has been found
            in the current search iteration.
        beta
            regularisation coefficent for the L1 term of the optimisation objective.
        gamma
            regularisation coefficient for the autoencoder term of the optimisation
            objective.
        feature_range
            range over which the features of the perturbed instances should be distributed.
        """
        classifier.train()
        autoencoder.train()

        self.classifier = classifier.forward_no_sm # obtain predictions scores from model BEFORE softmax
        self.autoencoder = autoencoder
        self.kappa = kappa
        self.c_init = c_init
        self.c_converge = c_converge
        self.beta = beta
        self.gamma = gamma

        self.iterations = iterations
        self.n_searches = n_searches
        self.learning_rate = learning_rate

        self.classifier_input_shape = classifier_input_shape
        self.autoencoder_input_shape = autoencoder_input_shape


    def explain(self, orig, mode="PN"):

        c = self.c_init

        orig = orig.view(28*28)

        self.best_loss = float('Inf')
        self.best_delta = None

        for search in range(self.n_searches):

            self.found_solution = False

            perturbation = torch.zeros(orig.shape) # perturbation 
            perturbation_s = torch.zeros(orig.shape, requires_grad=True) # perturbation slack variable

            # optimise for the slack variable y, with a square root decaying learning rate
            optim = torch.optim.SGD([perturbation_s], lr=self.learning_rate)

            for step in range(1, self.iterations + 1):

                optim.zero_grad()
                perturbation_s.requires_grad_(True)

                loss = self.loss_fn(orig, perturbation)

                # optimise for the slack variable, adjust lr
                loss.backward()
                optim.step()

                optim.param_groups[0]['lr'] = (self.learning_rate - 0.0)*(1 - step/self.iterations)**0.5

                perturbation_s.requires_grad_(False)

                cond1 = torch.gt(perturbation_s - orig, self.beta)
                cond2 = torch.le(torch.abs(perturbation_s - orig), self.beta)
                cond3 = torch.lt(perturbation_s- orig, -self.beta)
                upper = torch.min(perturbation_s - self.beta, torch.tensor(0.5))
                lower = torch.max(perturbation_s + self.beta, torch.tensor(-0.5))

                assign_perturbation = cond1.type(torch.float) * upper + cond2.type(torch.float) * orig + cond3.type(torch.float) * lower

                #ipdb.set_trace()

                cond4 = torch.gt(assign_perturbation - orig, 0).type(torch.float)
                cond5 = torch.le(assign_perturbation - orig, 0).type(torch.float)
                if mode == "PP":
                    assign_perturbation = cond5 * assign_perturbation + cond4 * orig
                elif mode == "PN":
                    assign_perturbation = cond4 * assign_perturbation + cond5 * orig

                assign_perturbation_s = assign_perturbation + step / (step + 3) * (assign_perturbation - perturbation)
                cond6 = torch.gt(assign_perturbation_s - orig, 0).type(torch.float)
                cond7 = torch.le(assign_perturbation_s - orig, 0).type(torch.float)

                if mode == "PP":
                    assign_perturbation_s = cond7 * assign_perturbation_s + cond6 * orig
                elif mode == "PN":
                    assign_perturbation_s = cond6 * assign_perturbation_s + cond7 * orig

                perturbation.data.copy_(assign_perturbation)
                perturbation_s.data.copy_(assign_perturbation_s)

                # check if the found delta solves the classification problem,
                # retain it if it is the most regularised solution
                if self.found_solution:
                    if loss < self.best_loss:
                        print("new best: {}".format(loss))
                        self.best_loss = loss
                        self.best_delta = perturbation.detach().clone()

                if not (step % 20):
                    print("search: {} iteration: {} c: {} loss: {:.2f} found optimum: {}".format(search, step, c, loss, self.found_solution))

            if found_optimum:
                c = (self.c_converge + c) / 2
            else:
                c *= 10

    def loss_fn(self, orig, perturbation_s):
        """
        loss function
        """
        # original classifier predictions before softmax
        orig_output = self.classifier(orig.view(*self.classifier_input_shape))
        
        # mask for the originally selected label (t_0)
        target_mask = torch.zeros(orig_output.shape)
        target_mask[torch.arange(orig_output.shape[0]), torch.argmax(orig_output)] = 1

        # mask for the originally non-selected labels (i =/= t_0)
        nontarget_mask = torch.ones(orig_output.shape) - target_mask

        # Optimise for image + delta, this is more stable
        delta_s = orig - perturbation_s

        if mode == "PP":

            target_lab_score = torch.max(target_mask * self.classifier(delta_s.view(*self.classifier_input_shape)))
            nontarget_lab_score = torch.max(nontarget_mask * self.classifier(delta_s.view(*self.classifier_input_shape)))
            
            loss_attack_s = c * torch.max(torch.tensor(0.), nontarget_lab_score_s - target_lab_score_s + self.kappa)
            
            loss_ae_dist_s = 0
            if callable(self.autoencoder):
                loss_ae_dist_s = self.gamma * (torch.norm(self.autoencoder(delta_s.view(*self.autoencoder_input_shape) + 0.5).view(*delta_s.shape) - 0.5 - delta_s))

        elif mode == "PN":

            target_lab_score_s = torch.max(target_mask * self.classifier(perturbation_s.view(*self.classifier_input_shape)))
            nontarget_lab_score_s = torch.max(nontarget_mask * self.classifier(perturbation_s.view(*self.classifier_input_shape)))
            
            loss_attack_s = c * torch.max(torch.tensor(0.), -nontarget_lab_score_s + target_lab_score_s + self.kappa)
            
            loss_ae_dist_s = 0
            if callable(self.autoencoder):
                loss_ae_dist_s = self.gamma * (torch.norm(self.autoencoder(perturbation_s.view(*self.autoencoder_input_shape) + 0.5).view(*delta_s.shape) - 0.5 - perturbation_s))

        if loss_attack_s.item() == 0:
            self.found_solution = True

        return loss_attack_s + torch.sum(delta_s ** 2) + loss_ae_dist_s
