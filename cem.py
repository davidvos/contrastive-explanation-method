""" This module implements the Contrastive Explanation Method in Pytorch.

Paper:  https://arxiv.org/abs/1802.07623
"""

import numpy as np
import torch

class ContrastiveExplanationMethod:
    
    def __init__(
        self,
        classifier,
        mode: str = "PP",
        autoencoder = None,
        kappa: float = 0.,
        const: float = 10.,
        beta: float = .1,
        gamma: float = 0.,
        feature_range: tuple = (-1e10, 1e10),
        iterations: int = 1000,
        n_searches: int = 9
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
        
        self.explain_model = explain_model
        self.mode = mode
        self.autoencoder = autoencoder
        self.kappa = kappa
        self.c_init = const
        self.c = const
        self.beta = beta
        self.gamma = gamma
        self.feature_range = feature_range
        self.iterations = iterations
        self.n_searches = n_searches
        
        self.delta = torch.zeros(orig_sample.shape)
        self.y = torch.zeros(orig_sample.shape)

        # projection space for binary datasets (X/x_0) for PN and (x_0) for PP
        if mode == "PN":
            self.pert_space = (torch.ones(original.shape) - orig_sample)
            self.pert_space /= torch.norm(self.pert_space, axis=1)
        elif mode == "PP":
            self.pert_space = orig_sample
            self.pert_space =/ torch.norm(self.pert_space, axis=1)

        # to keep track of whether in the current search the perturbation loss reached 0
        self.loss_reached_zero = False

        self.best_delta = None
        self.best_loss = float("Inf")
        self.prev_deltas = []
    
    def fista(self, orig_sample):
        """Fast Iterative Shrinkage Thresholding Algorithm implementation in pytorch
        
        Paper: https://doi.org/10.1137/080716542
        
        (Eq. 5) and (eq. 6) in https://arxiv.org/abs/1802.07623
        """

        if self.mode == "PP":
            delta_space = orig_sample.copy()
        elif self.mode == "PN":
            delta_space = torch.ones(orig_sample.shape) - orig_sample
        
        while stopping_condition:
            
            # See appendix A
            for _ in range(self.n_searches):
                
                for i in range(1, self.iterations + 1):

                    obj = (self.optimisation_obj(orig_sample)).sum()

                    if obj < self.best_loss:
                        self.best_delta = self.delta
                        self.best_loss = obj

                    obj.backward()
                    self.prev_deltas.append(self.delta.copy().detach())

                    # project onto subspace that contains our possible features. (eq. 5, 6)
                    self.delta = self.pert_space.dot(self.shrink(self.y - self.learning_rate * self.y.grad))
                    self.y = self.pert_space.dot((self.delta + i/(i + 3)(self.delta - self.prev_deltas[-1])))

                if self.loss_reached_zero:
                    self.c = self.c + self.c_init / 2
                else:
                    self.c *= 10

                    
    def shrink(self, z):
        """Element-wise shrinkage thresholding function.
        
        (Eq. 7) in https://arxiv.org/abs/1802.07623
        """
        zeros = torch.zeros(z.shape)
        z_min = z - self.beta
        z_plus = z + self.beta
        
        z_shrunk = z.copy()
        z_shrunk = torch.where(torch.abs(z) <= self.beta, zeros, z_shrunk)
        z_shrunk = torch.where(z > self.beta, z_min, z_shrunk)
        z_shrunk = torch.where(z < -self.beta, z_plus, z_shrunk)
        return z_shrunk
                        
    def optimisation_obj(self, orig_sample):
        """
        Optimisation objective for PN (eq. 1) and for PP (eq. 3).
        """
        
        obj = (
            self.c * self.loss_fn(orig_sample) +
            self.beta * torch.sum(torch.abs(self.y), axis=1) + # IS DIT GOED?
            torch.norm(self.y, axis=1) ** 2
        )
        if callable(self.autoencoder):
            if self.mode == "PN":
                obj + gamma * torch.norm(orig_sample + self.y - self.autoencoder(orig_sample + self.y).detach(), axis=1) ** 2
            elif self.mode == "PP":
                obj + gamma * torch.norm(self.y - self.autoencoder(self.y).detach(), axis=1) ** 2
        return obj

    def loss_fn(self, orig_sample):
        """
        Loss term f(x,d) for PN (eq. 2) and for PP (eq. 4).
        
        orig_sample
            the unperturbed original sample, batch size first.
        """
        
        orig_output = self.classifier(orig_sample)
        target_mask = torch.zeros(orig_output.shape)
        target_mask[torch.arange(orig_output.shape[0]), torch.argmax(orig_output, axis=1)] = 1
        nontarget_mask = torch.ones(orig_output.shape) - target_mask
        
        if self.mode == "PN":
            perturbation_loss = torch.max(
                torch.max(target_mask * self.classifier(orig_sample + self.y).detach(), axis=1) - 
                torch.max(nontarget_mask * self.classifier(orig_sample + self.y).detach(), axis=1),
                -self.kappa
            )
        elif self.mode == "PP":
            pert_output = self.classifier(self.y).detach()
            perturbation_loss = torch.max(
                torch.max(nontarget_mask * pert_output, axis=1) - torch.max(target_mask * pert_output, axis=1),
                -self.kappa
            )
        
        if perturbation_loss == -kappa:
            loss_reached_zero = True

        return perturbation_loss
    
