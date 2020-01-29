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
        learning_rate: float = 0.01,
        verbal: bool = False,
        print_every: int = 100,
        input_shape: tuple = (1, 28, 28),
        device: str = "cpu"
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
        classifier.eval()
        classifier.to(device)
        if autoencoder:
            autoencoder.eval()
            autoencoder.to(device)
        self.classifier = classifier.forward_no_sm
        self.autoencoder = autoencoder
        self.kappa = kappa
        self.c_converge = c_converge
        self.c_init = c_init
        self.beta = beta
        self.gamma = gamma

        self.iterations = iterations
        self.n_searches = n_searches
        self.learning_rate = learning_rate

        self.verbal = verbal
        self.input_shape = input_shape
        self.device = device
        self.print_every = print_every

    def explain(self, orig, mode="PN"):
        """
        Determine pertinents for a given input sample.

        orig
            The original input sample to find the pertinent for.
        mode
            Either "PP" for pertinent positives or "PN" for pertinent negatives.

        """
        if mode not in ["PN", "PP"]:
            raise ValueError(
                "Invalid mode. Please select either 'PP' or 'PN' as mode.")

        const = self.c_init
        step = 0

        orig = orig.view(*self.input_shape).to(self.device)

        best_loss = float("inf")
        best_delta = None

        orig_output = self.classifier(orig.view(-1, *self.input_shape))

        # mask for the originally selected label (t_0)
        target_mask = torch.zeros(orig_output.shape).to(self.device)
        target_mask[torch.arange(orig_output.shape[0]),
                    torch.argmax(orig_output)] = 1

        # mask for the originally non-selected labels (i =/= t_0)
        nontarget_mask = torch.ones(orig_output.shape).to(
            self.device) - target_mask

        for search in range(self.n_searches):

            found_optimum = False

            adv = torch.zeros(orig.shape).to(self.device)
            adv_s = torch.zeros(orig.shape).to(
                self.device).detach().requires_grad_(True)

            # optimise for the slack variable y, with a square root decaying learning rate
            optim = torch.optim.SGD([adv_s], lr=self.learning_rate)

            for step in range(1, self.iterations + 1):

                optim.zero_grad()
                adv_s.requires_grad_(True)

                ###############################################################
                #### Loss term f(x,d) for PN (eq. 2) and for PP (eq. 4). ######
                ###############################################################

                # Optimise for image + delta, this is more stable
                delta = orig - adv
                delta_s = orig - adv_s

                if mode == "PP":
                    img_to_enforce_label_score = self.classifier(
                        delta.view(-1, *self.input_shape))
                    img_to_enforce_label_score_s = self.classifier(
                        delta_s.view(-1, *self.input_shape))
                elif mode == "PN":
                    img_to_enforce_label_score = self.classifier(
                        adv.view(-1, *self.input_shape))
                    img_to_enforce_label_score_s = self.classifier(
                        adv_s.view(-1, *self.input_shape))

                # L2 regularisation term
                l2_dist_s = torch.sum(delta ** 2)

                target_lab_score_s = torch.max(
                    target_mask * img_to_enforce_label_score_s)
                nontarget_lab_score_s = torch.max(
                    nontarget_mask * img_to_enforce_label_score_s)

                if mode == "PP":
                    loss_attack_s = const * torch.max(torch.tensor(0.).to(
                        self.device), nontarget_lab_score_s - target_lab_score_s + self.kappa)
                elif mode == "PN":
                    loss_attack_s = const * torch.max(torch.tensor(0.).to(
                        self.device), -nontarget_lab_score_s + target_lab_score_s + self.kappa)

                loss_ae_dist_s = 0
                if mode == "PP" and callable(self.autoencoder):
                    loss_ae_dist_s = self.gamma * (torch.norm(self.autoencoder(
                        delta_s.view(-1, *self.input_shape) + 0.5).view(*self.input_shape) - 0.5 - delta_s)**2)
                elif mode == "PN" and callable(self.autoencoder):
                    loss_ae_dist_s = self.gamma * (torch.norm(self.autoencoder(
                        adv_s.view(-1, *self.input_shape) + 0.5).view(*self.input_shape) - 0.5 - adv_s)**2)

                loss_to_optimise = loss_attack_s + l2_dist_s + loss_ae_dist_s

                if loss_attack_s.item() == 0:
                    found_optimum = True

                # optimise for the slack variable, adjust lr
                loss_to_optimise.backward()
                optim.step()

                optim.param_groups[0]['lr'] = (
                    self.learning_rate - 0.0) * (1 - step/self.iterations) ** 0.5

                adv_s.requires_grad_(False)

                with torch.no_grad():

                    # Shrinkage thresholding function
                    zt = step / (step + 3)

                    cond1 = torch.gt(adv_s - orig, self.beta)
                    cond2 = torch.le(torch.abs(adv_s - orig), self.beta)
                    cond3 = torch.lt(adv_s - orig, -self.beta)
                    upper = torch.min(adv_s - self.beta,
                                      torch.tensor(0.5).to(self.device))
                    lower = torch.max(adv_s + self.beta,
                                      torch.tensor(-0.5).to(self.device))

                    assign_adv = cond1.type(
                        torch.float) * upper + cond2.type(torch.float) * orig + cond3.type(torch.float) * lower

                    # Apply projection and update steps
                    cond4 = torch.gt(assign_adv - orig, 0).type(torch.float)
                    cond5 = torch.le(assign_adv - orig, 0).type(torch.float)
                    if mode == "PP":
                        assign_adv = cond5 * assign_adv + cond4 * orig
                    elif mode == "PN":
                        assign_adv = cond4 * assign_adv + cond5 * orig

                    assign_adv_s = assign_adv + zt * (assign_adv - adv)
                    cond6 = torch.gt(assign_adv_s - orig, 0).type(torch.float)
                    cond7 = torch.le(assign_adv_s - orig, 0).type(torch.float)

                    if mode == "PP":
                        assign_adv_s = cond7 * assign_adv_s + cond6 * orig
                    elif mode == "PN":
                        assign_adv_s = cond6 * assign_adv_s + cond7 * orig

                    adv.data.copy_(assign_adv)
                    adv_s.data.copy_(assign_adv_s)

                    # check if the found delta solves the classification problem,
                    # retain it if it is the most regularised solution
                    if loss_attack_s.item() == 0:
                        if loss_to_optimise < best_loss:

                            best_loss = loss_to_optimise
                            best_delta = adv.detach().clone()

                            if self.verbal:
                                print("new best delta found with loss: {}".format(
                                    loss_to_optimise))

                    if self.verbal and not (step % self.print_every):
                        print("search: {} iteration: {} c: {} loss: {:.2f} found optimum: {}".format(
                            search, step, const, loss_to_optimise, found_optimum))

            if found_optimum:
                const = (self.c_converge + const) / 2
            else:
                const *= 10

        return best_delta
