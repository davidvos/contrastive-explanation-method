import torch.nn as nn
import torch
import numpy as np

class AEADEN:
	def __init__(self, model, mode, AE, batch_size, init_learning_rate, image_size, num_channels,
				  num_labels, binary_search_steps, max_iterations, kappa, c_init, beta, gamma, n_searches=9):

		assert mode in ["PP","PN"], "Model mode has to be set to either 'PP' or 'PN'"

		shape = (batch_size, image_size, image_size, num_channels)

		self.INIT_LR = init_learning_rate
		self.MAX_ITER = max_iterations
		self.BINARY_SEARCH_STEPS = binary_search_steps
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

		self.orig_img = torch.zeros(shape, dtype=torch.Float)
		self.delta_k = torch.zeros(shape, dtype=torch.Float) # adv_img
		self.y_k = torch.zeros(shape, dtype=torch.Float) # adv_img_s

		self.target_lab = torch.zeros((num_labels), dtype=torch.Float)
		self.const = torch.zeros(batch_size, dtype=torch.Float)
		
		self.global_step = torch.Tensor([0.0], dtype=torch.Float, requires_grad=False)


	def compute_CEM(self, img, label):
		self.orig_img = img
		# self.target_lab = torch.Tensor([label])

		orig_output = self.classifier(orig_sample.view(-1, 1, 28, 28))

		self.target_lab = torch.zeros(orig_output.shape)
        self.target_lab = target_mask[torch.arange(orig_output.shape[0]), torch.argmax(orig_output)] = 1

    	self.best_g = 0.0
		self.best_dist = float("inf")
		self.best_delta = None
		self.prev_deltas = []

		for s in range(self.n_searches):

			# to keep track of whether in the current search the perturbation loss reached 0
            self.pert_loss_reached_optimum = False

            optim = torch.optim.SGD([self.y_k], lr=self.INIT_LR)

			for iteration in range(1, self.MAX_ITER+1):
				optim.zero_grad()

				self.y_k.requires_grad_(True)

				self.compute_g()

				if self.loss_overall < self.best_g:
					self.best_delta = self.delta_k
					self.best_y = self.y_k
					self.best_g = self.loss_overall

				self.loss_to_opt.backward()
				optim.step()

				self.FISTA(step)
			

				lr = poly_lr_scheduler(init_lr=self.INIT_LR, cur_iter=self.global_step, end_learning_rate=0.0, lr_decay_iter=1, max_iter=self.MAX_ITER, power=0.5)
		    	adjust_optim(optim, lr)


	    	# adapt the perturbation loss coefficient
            if self.pert_loss_reached_optimum:
                self.c = (self.c + self.c_init) / 2
            else:
                self.c = self.c * 10





	def FISTA(self, step):
		self.zt = step / (step + 3)

		self.shrinkage()

		condition4 = torch.gt((self.delta_new - self.orig_img), 0.0).type(torch.Float) #line 73
		condition5 = torch.le((self.delta_new - self.orig_img), 0.0).type(torch.Float) #line 74

		if self.mode == "PP":
			self.delta_new = condition5 * self.delta_new + condition4 * self.orig_img # assign_adv_img
		elif self.mode == "PN":
			self.delta_new = condition4 * self.delta_new + condition5 * self.orig_img # assign_adv_img

		self.y_k_new = self.delta_new + self.zt * (self.delta_new - self.delta_k) # assign_adv_img_s

		condition6 = torch.gt((self.y_k_new - self.orig_img), 0).type(torch.Float)
		condition7 = torch.le((self.y_k_new - self.orig_img), 0).type(torch.Float)

		if self.mode == "PP":
			self.y_k_new = condition7 * self.y_k_new + condition6 * self.orig_img
		elif self.mode == "PN":
			self.y_k_new = condition6 * self.y_k_new + condition7 * self.orig_img

		self.delta_k = self.delta_new #UPDATE DELTA_K WITH K+1
		self.y_k.data.copy_(self.y_k_new) #UPDATE Y_K WITH K+1



	def shrinkage(self):
		condition1 = torch.gt((self.y_k - self.orig_img), self.beta).type(torch.Float) # cond1 line 66 checks if all entries of adv_img_s - orig img are bigger than beta
		condition2 = torch.le(abs(self.y_k - self.orig_img), self.beta).type(torch.Float) # cond2 line 67
		condition3 = torch.lt((self.y_k - self.orig_img), -self.beta).type(torch.Float)  #cond3 line 68

		upper = torch.min((self.y_k - self.beta), torch.Tensor([0.5], dtype=torch.Float)) #line 69
		lower = torch.max((self.y_k - self.beta), torch.Tensor([-0.5], dtype=torch.Float)) #line 70

		self.delta_new = condition1 * upper + condition2 * self.orig_img + condition3 * lower #line 71


    def fn_loss(self):
		self.delta_img = self.orig_img - self.delta_k
		self.y_img = self.orig_img - self.y_k

		if self.mode == "PP":
			self.img2label_score = self.classifier.predict(self.delta_img)
			self.img2label_y_score = self.classifier.predict(self.y_img)
		elif self.mode == "PN":
			self.img2label_score = self.classifier.predict(self.delta_k)
			self.img2label_y_score = self.classifier.predict(self.y_k)


		# compute the prob of true label class vs the next best class
		self.target_score = torch.sum(self.target_lab*self.img2label_score)
		target_score_s = torch.sum(self.target_lab*self.img2label_y_score)
		self.max_nontarget_score = torch.sum((1-self.target_lab)*self.img2label_score - (self.target_lab * 10000))
		max_nontarget_y_score = torch.sum((1-self.target_lab)*self.img2label_y_score - (self.target_lab * 10000))

		#PP loss formula (2)
		if self.mode == "PP":
			self.loss_alien_sample = torch.max(
            	self.max_nontarget_score - self.target_score + self.kappa, 
                torch.Tensor([0.0], dtype=torch.Float)
            )
            self.loss_alien_sample_s = torch.max(
                max_nontarget_y_score - target_score_s + self.kappa, 
                torch.Tensor([0.0], dtype=torch.Float)
            )
        #PN loss formula (3)
        elif self.mode == "PN":
        	self.loss_alien_sample = torch.max(
            	-self.max_nontarget_score + self.target_score + self.kappa, 
                torch.Tensor([0.0], dtype=torch.Float)
            )
            self.loss_alien_sample_s = torch.max(
                -max_nontarget_y_score + target_score_s + self.kappa, 
                torch.Tensor([0.0], dtype=torch.Float)
            )

    def compute_g(self):

    	self.fn_loss()

		# compute the l1, l2 and combined distance as part of formula 1 and 3
		self.l1_dist = torch.abs(self.delta_img).sum(dim=[1,2,3])
		self.l1_dist_s = torch.abs(self.y_img).sum(dim=[1,2,3])

		self.l2_dist = torch.pow(self.delta_img, 2).sum(dim=[1,2,3])
		self.l2_dist_s = torch.pow(self.y_img, 2).sum(dim=[1,2,3])

		self.dist = self.l2_dist + self.l1_dist * self.beta
		self.dist_s = self.l2_dist_s + self.l1_dist_s * self.beta

        #sum the losses
        self.loss_l1   = torch.sum(self.l1_dist)
        self.loss_l1_s = torch.sum(self.l1_dist_s)

        self.loss_l2   = torch.sum(self.l2_dist)
        self.loss_l2_s = torch.sum(self.l2_dist_s)

        self.loss_alien_sample   = torch.sum(self.c * self.loss_alien_sample)
        self.loss_alien_sample_s = torch.sum(self.c * self.loss_alien_sample_s)

        # last part of formula (1) and (3)
        if self.mode == "PP":
        	self.AE_loss   = self.gamma * torch.pow(torch.norm(self.AE(self.delta_img)-self.delta_img), 2)
        	self.AE_loss_s = self.gamma * torch.pow(torch.norm(self.AE(self.delta_img)-self.y_img), 2)

    	elif self.mode == "PN":
    		self.AE_loss   = self.gamma * torch.pow(torch.norm(self.AE(self.delta_k)-self.delta_k), 2)
        	self.AE_loss_s = self.gamma * torch.pow(torch.norm(self.AE(self.y_k)-self.y_k), 2)

    	self.loss_to_opt = self.loss_alien_sample_s + self.loss_l2_s + self.AE_loss_s
    	self.loss_overall = self.loss_alien_sample   + self.loss_l2   + self.AE_loss + self.beta * self.loss_l1


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