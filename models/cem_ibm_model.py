import torch.nn as nn
import torch
import numpy as np
import ipdb

class AEADEN:
    def __init__(
        self, 
        model, 
        mode: str = "PN", 
        AE = None, 
        batch_size: int = 1, 
        init_learning_rate: float = 0.1,
        image_size: int = 28, 
        num_channels: int = 1,
        num_labels: int = 10, 
        max_iterations: int = 1000, 
        kappa: float = .6,
        c_init: float = 0.1,
        beta: float = .1, 
        gamma: float = 100.,
        n_searches=9
    ):

        assert mode in ["PP","PN"], "Model mode has to be set to either 'PP' or 'PN' given: {}".format(mode)

        shape = (batch_size, num_channels, image_size, image_size)

        self.INIT_LR = init_learning_rate
        self.MAX_ITER = max_iterations
        self.BINARY_SEARCH_STEPS = n_searches
        self.batch_size = batch_size

        self.kappa = kappa
        self.c_init = c_init
        self.c = c_init
        self.beta = beta
        self.gamma = gamma
        self.mode = mode
        self.n_searches = n_searches

        self.classifier = model
        self.AE = AE

        self.orig_img = torch.zeros(shape, dtype=torch.double)
        self.adv_img = torch.zeros(shape, dtype=torch.double) # adv_img
        self.adv_img_s = torch.zeros(shape, dtype=torch.double) # adv_img_s

        self.target_lab = torch.zeros((num_labels), dtype=torch.double)
        self.const = torch.zeros(batch_size, dtype=torch.double)
        
        self.global_step = torch.tensor([0.0], dtype=torch.double, requires_grad=False)


    def compute_CEM(self, img):
        self.orig_img = img
        # print(f"img size {self.orig_img.shape} shape changed {self.orig_img.view(-1, 1, 28, 28).shape}")
        # self.target_lab = torch.Tensor([label])

        orig_output = self.classifier(self.orig_img.view(-1, 1, 28, 28))

        self.target_lab = torch.zeros(orig_output.shape).view(-1,10)
        self.target_lab[torch.arange(orig_output.shape[0]), torch.argmax(orig_output)] = 1

        self.best_g = 10000.0
        self.best_dist = float("inf")
        self.best_delta = None
        self.prev_deltas = []

        for s in range(self.n_searches):

            # to keep track of whether in the current search the perturbation loss reached 0
            self.pert_loss_reached_optimum = False

            optim = torch.optim.SGD([self.adv_img_s], lr=self.INIT_LR)

            for iteration in range(1, self.MAX_ITER+1):
                # print(iterationsion)
                optim.zero_grad()
 
                self.adv_img_s.requires_grad_(True)

                self.compute_g()

                if self.loss_overall < self.best_g:
                    self.best_delta = self.adv_img.clone()
                    self.best_y = self.adv_img_s
                    self.best_g = self.loss_overall

                if self.attack_s == 0:
                    self.pert_loss_reached_optimum = True

                self.loss_to_opt.backward()
                
                if not (iteration % 20):
                    print("search:{} iteration:{} lr:{:.2f} c value:{:.2f} loss: {:.2f} delta sum:{:.2f} optimum:{} y grad:{:.3f}".format(s, iteration, lr, self.c, self.attack_s.item(), self.adv_img.sum().item(), self.pert_loss_reached_optimum, self.adv_img_s.grad.sum()))


                optim.step()

                self.adv_img_s.requires_grad_(False)

                self.FISTA(iteration)


            

                lr = poly_lr_scheduler(init_lr=self.INIT_LR, cur_iter=iteration, end_learning_rate=0.0, lr_decay_iter=1, max_iter=self.MAX_ITER, power=0.5)
                adjust_optim(optim, lr)


            # adapt the perturbation loss coefficient
            if self.pert_loss_reached_optimum:
                print("hallo")
                self.c = (self.c + self.c_init) / 2
            else:
                print("doei")
                self.c = self.c * 10





    def FISTA(self, step):
        self.zt = step / (step + 3)

        self.shrinkage()

        condition4 = torch.gt(torch.sub(self.assign_adv_img, self.orig_img), 0.0).type(torch.double) #line 73
        condition5 = torch.le(torch.sub(self.assign_adv_img, self.orig_img), 0.0).type(torch.float) #line 74

        if self.mode == "PP":
            self.assign_adv_img = torch.mul(condition5, self.assign_adv_img) + torch.mul(condition4, self.orig_img) # assign_adv_img
        elif self.mode == "PN":
            self.assign_adv_img = torch.mul(condition4, self.assign_adv_img) + torch.mul(condition5, self.orig_img) # assign_adv_img

        self.assign_adv_img_s = self.assign_adv_img + torch.mul(self.zt, (self.assign_adv_img - self.adv_img)) # assign_adv_img_s

        condition6 = torch.gt(torch.sub(self.assign_adv_img_s, self.orig_img), 0).type(torch.float)
        condition7 = torch.le(torch.sub(self.assign_adv_img_s, self.orig_img), 0).type(torch.float)

        if self.mode == "PP":
            self.assign_adv_img_s = torch.mul(condition7, self.assign_adv_img_s) + torch.mul(condition6, self.orig_img)
        elif self.mode == "PN":
            self.assign_adv_img_s = torch.mul(condition6, self.assign_adv_img_s) + torch.mul(condition7, self.orig_img)

        # print(f"condition shape: {condition7.shape} and sum shape: {(condition7 * self.assign_adv_img_s).shape}")

        # print(f"y_k shape: {self.adv_img_s.shape} and new shape: {self.assign_adv_img_s.shape}")
        self.adv_img = self.assign_adv_img #UPDATE DELTA_K WITH K+1
        self.adv_img_s.data.copy_(self.assign_adv_img_s) #UPDATE Y_K WITH K+1


    def shrinkage(self):
        condition1 = torch.gt(torch.sub(self.adv_img_s, self.orig_img), self.beta).type(torch.float) # cond1 line 66 checks if all entries of adv_img_s - orig img are bigger than beta
        condition2 = torch.le(abs(torch.sub(self.adv_img_s, self.orig_img)), self.beta).type(torch.float) # cond2 line 67
        condition3 = torch.lt(torch.sub(self.adv_img_s, self.orig_img), -self.beta).type(torch.float)  #cond3 line 68

        upper = torch.min(torch.sub(self.adv_img_s, self.beta), torch.tensor([0.5], dtype=torch.double)) #line 69
        lower = torch.max(torch.add(self.adv_img_s, self.beta), torch.tensor([-0.5], dtype=torch.double)) #line 70

        self.assign_adv_img = torch.mul(condition1, upper) + torch.mul(condition2, self.orig_img) + torch.mul(condition3, lower) #line 71
        # print(f"shape new delta: {self.assign_adv_img.shape} condition shape: {condition1.shape}")


    def fn_loss(self):
        self.delta_img = self.orig_img.view(-1,1,28,28) - self.adv_img
        self.delta_img_s = self.orig_img.view(-1,1,28,28) - self.adv_img_s
        # print(f"img size {self.delta_img.shape} orig {self.orig_img.shape} orig view {self.orig_img.view(-1,1,28,28).shape} delta: {self.adv_img.shape}")

        if self.mode == "PP":
            self.img2label_score   = self.classifier.forward_no_sm(self.delta_img.view(-1,1,28,28).float())
            self.img2label_y_score = self.classifier.forward_no_sm(self.delta_img_s.view(-1,1,28,28).float())
        elif self.mode == "PN":
            self.img2label_score   = self.classifier.forward_no_sm(self.adv_img.view(-1,1,28,28).float())
            self.img2label_y_score = self.classifier.forward_no_sm(self.adv_img_s.view(-1,1,28,28).float())


        # compute the prob of true label class vs the next best class
        self.target_score        = torch.sum(self.target_lab*self.img2label_score, dim=1)
        target_score_s           = torch.sum(self.target_lab*self.img2label_y_score, dim=1)
        self.max_nontarget_score = torch.max((1-self.target_lab)*self.img2label_score - (self.target_lab * 10000), dim=1)[0]
        max_nontarget_y_score    = torch.max((1-self.target_lab)*self.img2label_y_score - (self.target_lab * 10000), dim=1)[0]

        if self.pert_loss_reached_optimum:
            print(f" max non target: {self.max_nontarget_score} \n target: {self.target_score}\n target s: {target_score_s}\n max non target y: {max_nontarget_y_score}\n")
            piemrld()

        # print(f"value max non: {self.max_nontarget_score} value max non y {max_nontarget_y_score}")
        #PP loss formula (2)
        if self.mode == "PP":
            self.attack = torch.max(
                self.max_nontarget_score - self.target_score + self.kappa, 
                torch.tensor([0.0], dtype=torch.float)
            )
            self.attack_s = torch.max(
                max_nontarget_y_score - target_score_s + self.kappa, 
                torch.tensor([0.0], dtype=torch.float)
            )
        #PN loss formula (3)
        elif self.mode == "PN":
            self.attack = torch.max(
                -self.max_nontarget_score + self.target_score + self.kappa, 
                torch.tensor([0.0], dtype=torch.float)
            )
            self.attack_s = torch.max(
                -max_nontarget_y_score + target_score_s + self.kappa, 
                torch.tensor([0.0], dtype=torch.float)
            )

    def compute_g(self):

        self.fn_loss()

        # compute the l1, l2 and combined distance as part of formula 1 and 3
        self.l1_dist   = torch.sum(torch.abs(self.delta_img), dim=[1,2,3])
        self.l1_dist_s = torch.sum(torch.abs(self.delta_img_s), dim=[1,2,3])

        self.l2_dist   = torch.sum(torch.pow(self.delta_img, 2), dim=[1,2,3])
        self.l2_dist_s = torch.sum(torch.pow(self.delta_img_s, 2), dim=[1,2,3])

        self.dist      = self.l2_dist + torch.mul(self.l1_dist, self.beta)
        self.dist_s    = self.l2_dist_s + torch.mul(self.l1_dist_s, self.beta)

        #sum the losses
        self.loss_l1   = torch.sum(self.l1_dist)
        self.loss_l1_s = torch.sum(self.l1_dist_s)

        self.loss_l2   = torch.sum(self.l2_dist)
        self.loss_l2_s = torch.sum(self.l2_dist_s)

        self.attack   = torch.sum(self.c * self.attack)
        self.attack_s = torch.sum(self.c * self.attack_s)

        # last part of formula (1) and (3)
        self.AE.double()
        if self.mode == "PP":
            self.AE_loss   = self.gamma * torch.pow(torch.norm(self.AE(self.delta_img.view(-1,1,28,28)+0.5)-0.5-self.delta_img), 2)
            self.AE_loss_s = self.gamma * torch.pow(torch.norm(self.AE(self.delta_img.view(-1,1,28,28)+0.5)-0.5-self.delta_img_s), 2)

        elif self.mode == "PN":
            self.AE_loss   = self.gamma * torch.pow(torch.norm(self.AE(self.adv_img.view(-1,1,28,28)+0.5)-0.5-self.adv_img), 2)
            self.AE_loss_s = self.gamma * torch.pow(torch.norm(self.AE(self.adv_img_s.view(-1,1,28,28)+0.5)-0.5-self.adv_img_s), 2)

        self.loss_to_opt = self.attack_s + self.loss_l2_s + self.AE_loss_s
        self.loss_overall = self.attack   + self.loss_l2   + self.AE_loss + torch.mul(self.beta, self.loss_l1)


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
    # print("hallo ", lr.detach().data)
    optimizer.param_groups[0]['lr'] = lr#.detach().data