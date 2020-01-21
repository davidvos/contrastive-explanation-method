""" This module implements the Contrastive Explanation Method in Pytorch.

Paper:  https://arxiv.org/abs/1802.07623
"""

import numpy as np
import torch
import sys
import ipdb

class ContrastiveExplanationMethod:
    
    def __init__(
        self,
        classifier,
        autoencoder = None,
        kappa: float = .6,
        const: float = 0.1,
        beta: float = .1,
        gamma: float = 100.,
        feature_range: tuple = (-1e10, 1e10),
        iterations: int = 1000,
        n_searches: int = 9,
        learning_rate: float = 0.1,
        batch: bool = False,
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
        const
            initial regularisation coefficient for the attack loss term.
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
        self.classifier = classifier.forward_no_sm
        self.autoencoder = autoencoder
        self.kappa = kappa
        self.c_init = const
        self.c = const
        self.beta = beta
        self.gamma = gamma
        self.feature_range = feature_range
        self.iterations = iterations
        self.n_searches = n_searches
        self.learning_rate = learning_rate

        # if input is batch (as opposed to single sample), reduce dimensions along second axis, otherwise reduce along first axis
        self.reduce_dim = int(batch)

    def fista(self, orig_img, mode="PN"):
        """Fast Iterative Shrinkage Thresholding Algorithm implementation in pytorch
        
        Paper: https://doi.org/10.1137/080716542
        
        (Eq. 5) and (eq. 6) in https://arxiv.org/abs/1802.07623
        """
        # initialise search values
        self.mode = mode

        orig_img = orig_img.view(28*28)
        perturb_init = torch.zeros(orig_img.shape)

        self.best_delta = None
        self.best_loss = float("Inf")

        self.c = 10

        # See appendix A
        for s in range(self.n_searches):

            # to keep track of whether in the current search the perturbation loss reached 0
            self.pert_loss_reached_optimum = False

            # initialise values for a new search
            delta = torch.zeros(orig_img.shape)
            y = torch.zeros(orig_img.shape, requires_grad=True)

            # optimise for the slack variable y, with a square root decaying learning rate
            optim = torch.optim.SGD([y], lr=self.learning_rate)

            for i in range(self.iterations):

                # Reset the computational graph, otherwise we get a multiple backward passes error
                optim.zero_grad()

                y.requires_grad_(True)

                # calculate loss as per (eq. 1, 3)
                loss = self.loss_fn(orig_img, y).sum()
                loss.backward()
                
                optim.step()
                lr = poly_lr_scheduler(init_lr=self.learning_rate, cur_iter=i, end_learning_rate=0.0, lr_decay_iter=1, max_iter=self.iterations, power=0.5)
                adjust_optim(optim, lr)

                y.requires_grad_(False)

                # store previous delta
                prev_delta = delta.clone().detach()

                if self.pert_loss.item() == 0:
                    if loss < self.best_loss:
                        print("NEW BEST: {} - C: {}".format(loss.item(), self.c))
                        self.best_delta = delta.clone().detach()
                        self.best_loss = loss.item()
                        self.best_c = self.c
                        self.best_pert_loss = self.pert_loss.clone().detach()

                if not (i % 20):
                    print("search:{} iteration:{} lr:{:.2f} c value:{:.2f} loss: {:.2f} delta sum:{:.2f} optimum:{} y grad:{:.3f}".format(s, i, lr, self.c, loss.item(), delta.sum().item(), self.pert_loss_reached_optimum, y.grad.sum()))

                # optimise for the sample + y since this is more stable
                y.data.copy_(self.shrink(y - orig_img) + orig_img)

                # perform the first projection step
                if self.mode == "PN":
                    delta.data.copy_(torch.where(y > orig_img, y, orig_img))
                elif self.mode == "PP":
                    delta.data.copy_(torch.where(y <= orig_img, y, orig_img))

                delta_momentum = (delta + i/(i + 3)*(delta - prev_delta))

                # perform second projection step
                if self.mode == "PN":
                    y.data.copy_(torch.where(delta_momentum > orig_img, delta_momentum, orig_img))
                elif self.mode == "PP":
                    y.data.copy_(torch.where(delta_momentum <= orig_img, delta_momentum, orig_img))

            # adapt the perturbation loss coefficient
            if self.pert_loss_reached_optimum:
                self.c = (self.c + self.c_init) / 2
            else:
                self.c = self.c * 10

    def shrink(self, z):
        """Element-wise shrinkage thresholding function.
        
        (Eq. 7) in https://arxiv.org/abs/1802.07623
        """
        zeros = torch.zeros(z.shape)
        z_min = z - self.beta
        z_plus = z + self.beta
        
        z_shrunk = z.clone()
        z_shrunk = torch.where(torch.abs(z) <= self.beta, zeros, z_shrunk)
        z_shrunk = torch.where(z > self.beta, z_min, z_shrunk)
        z_shrunk = torch.where(z < -self.beta, z_plus, z_shrunk)
        z_shrunk = torch.where(z_shrunk > 0.5, torch.tensor(0.5), z_shrunk)
        z_shrunk = torch.where(z_shrunk < -0.5, torch.tensor(-0.5), z_shrunk)
        return z_shrunk

    def loss_fn(self, orig_img, y):
        """
        Optimisation objective for PN (eq. 1) and for PP (eq. 3).
        """
        obj = (
            self.c * self.perturbation_loss(orig_img, y) +
            torch.norm(y) ** 2
        )

        if callable(self.autoencoder):
            if self.mode == "PN":
                obj += self.gamma * torch.norm(orig_img + y - 0.5 - self.autoencoder((orig_img + y - 0.5).view(-1, 1, 28, 28)).view(28*28)) ** 2 # TEMP FIX model trained on 0 to 1 range
            elif self.mode == "PP":
                obj += self.gamma * torch.norm(y - 0.5 - self.autoencoder(y.view(-1, 1, 28, 28) - 0.5).view(28*28)) ** 2  # TEMP FIX

        #ipdb.set_trace()
        return obj

    def perturbation_loss(self, orig_img, y):
        """
        Loss term f(x,d) for PN (eq. 2) and for PP (eq. 4).
        
        orig_img
            the unpertrbed original sample, batch size first.
        """
        
        orig_output = self.classifier(orig_img.view(-1, 1, 28, 28))

        # mask for the originally selected label (t_0)
        target_mask = torch.zeros(orig_output.shape)
        target_mask[torch.arange(orig_output.shape[0]), torch.argmax(orig_output)] = 1

        # mask for the originally non-selected labels (i =/= t_0)
        nontarget_mask = torch.ones(orig_output.shape) - target_mask

        if self.mode == "PN":
            pert_output = self.classifier((orig_img + y).view(-1, 1, 28, 28))
            perturbation_loss = torch.max(
                torch.max(target_mask * pert_output) -
                torch.max(nontarget_mask * pert_output) + self.kappa,
                torch.tensor(0.)
            )
        elif self.mode == "PP":
            pert_output = self.classifier(y.view(-1, 1, 28, 28))
            perturbation_loss = torch.max(
                torch.max(nontarget_mask * pert_output) -
                torch.max(target_mask * pert_output) + self.kappa,
                torch.tensor(0.)
            )


        #ipdb.set_trace()
        self.pert_loss = perturbation_loss

        if perturbation_loss.item() == 0:
            self.pert_loss_reached_optimum = True

        return perturbation_loss

def poly_lr_scheduler(init_lr, cur_iter, lr_decay_iter=1,
                      max_iter=100, end_learning_rate=0.0, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    lr = (init_lr-end_learning_rate)*(1 - cur_iter/max_iter)**power + end_learning_rate

    return lr


def adjust_optim(optimizer, lr):
    optimizer.param_groups[0]['lr'] = lr