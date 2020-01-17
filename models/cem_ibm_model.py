import torch.nn as nn
import torch
import numpy as np

class AEADEN:
	def __init__(self, model, mode, AE, batch_size, init_learning_rate, image_size, num_channels,
				  num_labels, binary_search_steps, max_iterations, kappa, c_init, beta, gamma):

		assert mode in ["PP","PN"], "Model mode has to be set to either 'PP' or 'PN'"

		shape = (batch_size, image_size, image_size, num_channels)

		self.INIT_LR = init_learning_rate
		self.MAX_ITER = max_iterations
		self.BINARY_SEARCH_STEPS = binary_search_steps
		self.batch_size = batch_size

		self.kappa = kappa
		self.c_init = c_init
		self.beta = beta
		self.gamma = gamma
		self.mode = mode

		self.AE = AE

		self.orig_img = torch.zeros(shape, dtype=torch.Float)
		self.adv_img = torch.zeros(shape, dtype=torch.Float) # delta_k
		self.adv_img_s = torch.zeros(shape, dtype=torch.Float) # y_k
		self.target_lab = torch.zeros((batch_size, num_labels), dtype=torch.Float)
		self.const = torch.zeros(batch_size, dtype=torch.Float)
		self.global_step = torch.Tensor([0.0], dtype=torch.Float, requires_grad=False)


	def FISTA(self):
		self.zt = self.global_step / (self.global_step + torch.Tensor([3.0], dtype=torch.Float))

		condition4 = torch.gt((self.delta_1 - self.orig_img), 0.0).type(torch.Float) #line 73
		condition5 = torch.le((self.delta_1 - self.orig_img), 0.0).type(torch.Float) #line 74

		if self.mode == "PP":
			self.delta_new = condition5 * self.delta_1 + condition4 * self.orig_img # assign_adv_img
		elif self.mode == "PN":
			self.delta_new = condition4 * self.delta_1 + condition5 * self.orig_img # assign_adv_img

		self.assign_adv_img_s = self.delta_new + self.zt * (self.delta_new - self.adv_img) # y_k_new

		condition6 = torch.gt((self.assign_adv_img_s - self.orig_img), 0).type(torch.Float)
		condition7 = torch.le((self.assign_adv_img_s - self.orig_img), 0).type(torch.Float)

		if self.mode == "PP":
			self.assign_adv_img_s = condition7 * self.assign_adv_img_s + condition6 * self.orig_img
		elif self.mode == "PN":
			self.assign_adv_img_s = condition6 * self.assign_adv_img_s + condition7 * self.orig_img

		self.adv_img = self.delta_new #UPDATE DELTA_K WITH K+1
		self.adv_img_s = self.assign_adv_img_s #UPDATE Y_K WITH K+1


	def shrinkage(self):
		condition1 = torch.gt((self.adv_img_s - self.orig_img), self.beta).type(torch.Float) # cond1 line 66 checks if all entries of adv_img_s - orig img are bigger than beta
		condition2 = torch.le(abs(self.adv_img_s - self.orig_img), self.beta).type(torch.Float) # cond2 line 67
		condition3 = torch.lt((self.adv_img_s - self.orig_img), -self.beta).type(torch.Float)  #cond3 line 68

		upper = torch.min((self.adv_img_s - self.beta), torch.Tensor([0.5], dtype=torch.Float)) #line 69
		lower = torch.max((self.adv_img_s - self.beta), torch.Tensor([-0.5], dtype=torch.Float)) #line 70

		self.delta_new = condition1 * upper + condition2 * self.orig_img + condition3 * lower #line 71



	def compute_CEM(self):
		self.delta_img = self.orig_img - self.adv_img
		self.y_img = self.orig_img - self.adv_img_s

		if self.mode == "PP":
			self.img2label_score = model.predict(self.delta_img)
			self.img2label_y_score = model.predict(self.y_img)
		elif self.mode == "PN":
			self.img2label_score = model.predict(self.adv_img)
			self.img2label_y_score = model.predict(self.adv_img_s)

		# compute the l1, l2 and combined distance as part of formula 1 and 3
		self.l1_dist = torch.abs(self.delta_img).sum(dim=[1,2,3])
		self.l1_dist_s = torch.abs(self.y_img).sum(dim=[1,2,3])

		self.l2_dist = torch.pow(self.delta_img, 2).sum(dim=[1,2,3])
		self.l2_dist_s = torch.pow(self.y_img, 2).sum(dim=[1,2,3])

		self.dist = self.l2_dist + self.l1_dist * self.beta
		self.dist_s = self.l2_dist_s + self.l1_dist_s * self.beta

		# compute the prob of true label class vs the next best class
		self.target_score = torch.sum(self.target_lab*self.img2label_score, dim=1)
		target_score_s = torch.sum(self.target_lab*self.img2label_y_score, dim=1)
		self.max_nontarget_score = torch.sum((1-self.target_lab)*self.img2label_score - (self.target_lab * 10000), dim=1)
		max_nontarget_y_score = torch.sum((1-self.target_lab)*self.img2label_y_score - (self.target_lab * 10000), dim=1)

		#PP loss formula (2)
		if self.mode == "PP":
			loss = torch.max(
            	self.max_nontarget_score - self.target_score + self.kappa, 
                torch.Tensor([0.0], dtype=torch.Float)
            )
            loss_y = torch.max(
                max_nontarget_y_score - target_score_s + self.kappa, 
                torch.Tensor([0.0], dtype=torch.Float)
            )
        #PN loss formula (3)
        elif self.mode == "PN":
        	loss = torch.max(
            	-self.max_nontarget_score + self.target_score + self.kappa, 
                torch.Tensor([0.0], dtype=torch.Float)
            )
            loss_y = torch.max(
                -max_nontarget_y_score + target_score_s + self.kappa, 
                torch.Tensor([0.0], dtype=torch.Float)
            )

        #sum the losses
        self.loss_l1   = torch.sum(self.l1_dist)
        self.loss_l1_s = torch.sum(self.l1_dist_s)

        self.loss_l2   = torch.sum(self.l2_dist)
        self.loss_l2_s = torch.sum(self.l2_dist_s)

        self.loss_alien_sample   = torch.sum(self.c * loss)
        self.loss_alien_sample_s = torch.sum(self.c * loss_y)

        # last part of formula (1) and (3)
        if self.mode == "PP":
        	self.AE_loss   = self.gamma * torch.pow(torch.norm(self.AE(self.delta_img)-self.delta_img), 2)
        	self.AE_loss_s = self.gamma * torch.pow(torch.norm(self.AE(self.delta_img)-self.y_img), 2)

    	elif self.mode == "PN":
    		self.AE_loss   = self.gamma * torch.pow(torch.norm(self.AE(self.adv_img)-self.adv_img), 2)
        	self.AE_loss_s = self.gamma * torch.pow(torch.norm(self.AE(self.adv_img_s)-self.adv_img_s), 2)

    	self.loss_to_opt = self.loss_alien_sample_s + self.loss_l2_s + self.AE_loss_s
    	self.loss_overall = self.loss_alien_sample   + self.loss_l2   + self.AE_loss + self.beta * self.loss_l1

    	optimizer = torch.optim.SGD(model.parameters(), lr=self.INIT_LR) #moet nog anders, verder vanaf line 140
    	lr = poly_lr_scheduler(init_lr=self.INIT_LR, cur_iter=self.global_step, end_learning_rate=0.0, lr_decay_iter=1, max_iter=self.MAX_ITER, power=0.5)


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