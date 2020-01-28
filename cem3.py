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
        kappa: float = .6,
        const: float = 0.1,
        c_init: float = 10.0,
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
        self.c = c_init
        self.beta = beta
        self.gamma = gamma
        self.feature_range = feature_range
        self.iterations = iterations
        self.n_searches = n_searches
        self.learning_rate = learning_rate

        # if input is batch (as opposed to single sample), reduce dimensions along second axis, otherwise reduce along first axis
        self.reduce_dim = int(batch)


    def fista(self, orig, mode="PN"):

        if mode not in ["PN", "PP"]:
            raise ValueError("Invalid mode. Please select either 'PP' or 'PN' as mode.")

        const = self.c
        step = 0

        orig = orig.view(28*28)

        self.best_loss = float("inf")
        self.best_delta = None

        orig_output = self.classifier(orig.view(-1, 1, 28, 28))

        # mask for the originally selected label (t_0)
        target_mask = torch.zeros(orig_output.shape)
        target_mask[torch.arange(orig_output.shape[0]), torch.argmax(orig_output)] = 1

        # mask for the originally non-selected labels (i =/= t_0)
        nontarget_mask = torch.ones(orig_output.shape) - target_mask

        for search in range(self.n_searches):

            found_optimum = False

            # adv = torch.zeros(orig.shape) # delta_k
            # adv_s = torch.zeros(orig.shape, requires_grad=True) # y_k

            adv = torch.zeros(orig.shape)
            adv_s = torch.zeros(orig.shape, requires_grad=True)
            adv_s.requires_grad_(True)

            # optimise for the slack variable y, with a square root decaying learning rate
            optim = torch.optim.SGD([adv_s], lr=self.learning_rate)

            for step in range(1, self.iterations + 1):

                optim.zero_grad()
                adv_s.requires_grad_(True)

                # Optimise for image + delta, this is more stable
                delta = orig - adv
                delta_s = orig - adv_s

                if mode == "PP":
                    img_to_enforce_label_score = self.classifier(delta.view(-1, 1, 28, 28))
                    img_to_enforce_label_score_s = self.classifier(delta_s.view(-1, 1, 28, 28))
                elif mode == "PN":
                    img_to_enforce_label_score = self.classifier(adv.view(-1, 1, 28, 28))
                    img_to_enforce_label_score_s = self.classifier(adv_s.view(-1, 1, 28, 28))

                # regularisation terms
                l2_dist = torch.sum(delta ** 2)
                l2_dist_s = torch.sum(delta ** 2)
                l1_dist = torch.abs(delta).sum()
                l1_dist_s = torch.abs(delta).sum()

                en_dist = l2_dist + self.beta * l1_dist
                en_dist_s = l2_dist_s + self.beta * l1_dist_s

                target_lab_score = torch.max(target_mask * img_to_enforce_label_score)
                target_lab_score_s = torch.max(target_mask * img_to_enforce_label_score_s)
                nontarget_lab_score = torch.max(nontarget_mask * img_to_enforce_label_score)
                nontarget_lab_score_s = torch.max(nontarget_mask * img_to_enforce_label_score_s)

                if mode == "PP":
                    loss_attack = const * torch.max(torch.tensor(0.), nontarget_lab_score - target_lab_score + self.kappa)
                    loss_attack_s = const * torch.max(torch.tensor(0.), nontarget_lab_score_s - target_lab_score_s + self.kappa)
                elif mode == "PN":
                    loss_attack = const * torch.max(torch.tensor(0.), -nontarget_lab_score + target_lab_score + self.kappa)
                    loss_attack_s = const * torch.max(torch.tensor(0.), -nontarget_lab_score_s + target_lab_score_s + self.kappa)

                if mode == "PP" and callable(self.autoencoder):
                    loss_ae_dist = self.gamma * (torch.norm(self.autoencoder(delta.view(-1,1,28,28) + 0.5).view(28*28) - 0.5 - delta)**2)
                    loss_ae_dist_s = self.gamma * (torch.norm(self.autoencoder(delta_s.view(-1,1,28,28) + 0.5).view(28*28) - 0.5 - delta_s)**2)
                elif mode == "PN" and callable(self.autoencoder):
                    loss_ae_dist = self.gamma * (torch.norm(self.autoencoder(adv.view(-1,1,28,28) + 0.5).view(28*28) - 0.5 - adv)**2)
                    loss_ae_dist_s = self.gamma * (torch.norm(self.autoencoder(adv_s.view(-1,1,28,28) + 0.5).view(28*28) - 0.5 - adv_s)**2)

                #ipdb.set_trace()

                loss_to_optimise = loss_attack_s + l2_dist_s + loss_ae_dist_s
                loss_for_display_purposes = loss_attack + l2_dist + loss_ae_dist

                #ipdb.set_trace()

                if loss_attack_s.item() == 0:
                    found_optimum = True

                # optimise for the slack variable, adjust lr
                loss_to_optimise.backward()
                optim.step()
                
                lr = poly_lr_scheduler(init_lr=self.learning_rate, cur_iter=step, end_learning_rate=0.0, lr_decay_iter=1, max_iter=self.iterations, power=0.5)
                optim.param_groups[0]['lr'] = lr

                adv_s.requires_grad_(False)

                # fast iterative shrinkage thresholding function
                zt = step/(step+3)

                cond1 = torch.gt(adv_s - orig, self.beta)
                cond2 = torch.le(torch.abs(adv_s - orig), self.beta)
                cond3 = torch.lt(adv_s- orig, -self.beta)
                upper = torch.min(adv_s - self.beta, torch.tensor(0.5))
                lower = torch.max(adv_s + self.beta, torch.tensor(-0.5))

                assign_adv = cond1.type(torch.float) * upper + cond2.type(torch.float) * orig + cond3.type(torch.float) * lower

                #ipdb.set_trace()

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
                if loss_attack.item() == 0:
                    found_optimum = True
                    if loss_to_optimise < self.best_loss:
                        print("new best: {}".format(loss_to_optimise))
                        self.best_loss = loss_to_optimise
                        self.best_delta = adv.detach().clone()

                if not (step % 1):
                    print("search: {} iteration: {} c: {} loss: {:.2f} found optimum: {}".format(search, step, const, loss_to_optimise, found_optimum))

            if found_optimum:
                const = (self.c_init + const) / 2
            else:
                const *= 10


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