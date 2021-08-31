
import numpy as np
	
class CrowdsBinaryAggregator():

	def __init__(self, model, data_train, answers, batch_size = 16, alpha_prior = 1.0, beta_prior = 1.0):
		self.model = model
		self.data_train = data_train
		self.answers = answers
		self.batch_size = batch_size
		self.alpha_prior = alpha_prior
		self.beta_prior = beta_prior
		self.n_train = answers.shape[0]
		self.num_annotators = answers.shape[1]

		# initialize annotators as reliable (almost perfect)
		self.alpha = 0.5 * np.ones(self.num_annotators)
		self.beta = 0.5 * np.ones(self.num_annotators)

		# initialize estimated ground truth with majority voting
		self.ground_truth_est = np.zeros((self.n_train, 2))
		for i in range(self.n_train):
			votes = np.zeros(self.num_annotators)
			for r in range(self.num_annotators):
				if answers[i,r] != -1:
					votes[answers[i,r]] += 1
			self.ground_truth_est[i,np.argmax(votes)] = 1


	def e_step(self):
		print("E-step")
		for i in range(self.n_train):
			a = 1.0
			b = 1.0
			for r in range(self.num_annotators):
				if self.answers[i,r] != -1:
					if self.answers[i,r] == 1:
						a *= self.alpha[r]
						b *= (1-self.beta[r])
					elif self.answers[i,r] == 0:
						a *= (1-self.alpha[r])
						b *= self.beta[r]
					else:
						raise Exception()
			mu = (a * self.ground_truth_est[i,1]) / (a * self.ground_truth_est[i,1] + b * self.ground_truth_est[i,0])
			self.ground_truth_est[i,1] = mu
			self.ground_truth_est[i,0] = 1.0 - mu

		return self.ground_truth_est


	def m_step(self, epochs=1):
		print("M-step")
		hist = self.model.fit(self.data_train, self.ground_truth_est, epochs=epochs, shuffle=True, batch_size=self.batch_size, verbose=0) 
		print(("loss:", hist.history["loss"][-1]))
		self.ground_truth_est = self.model.predict(self.data_train)

		self.alpha = self.alpha_prior*np.ones(self.num_annotators)
		self.beta = self.beta_prior*np.ones(self.num_annotators)
		for r in range(self.num_annotators):
			alpha_norm = 0.0
			beta_norm = 0.0
			for i in range(self.n_train):
				if self.answers[i,r] != -1:
					alpha_norm += self.ground_truth_est[i,1]
					beta_norm += self.ground_truth_est[i,0]
					if self.answers[i,r] == 1:
						self.alpha[r] += self.ground_truth_est[i,1]
					elif self.answers[i,r] == 0:
						self.beta[r] += self.ground_truth_est[i,0]
					else:
						raise Exception()
			self.alpha[r] /= alpha_norm
			self.beta[r] /= beta_norm
		
		return self.model, self.alpha, self.beta
		

class CrowdsCategoricalAggregator():

	def __init__(self, model, data_train, answers, batch_size = 16, pi_prior = 1.0):
		self.model = model
		self.data_train = data_train
		self.answers = answers
		self.batch_size = batch_size
		self.pi_prior = pi_prior
		self.n_train = answers.shape[0]
		self.num_classes = np.max(answers) + 1
		self.num_annotators = answers.shape[1]

		# initialize annotators as reliable (almost perfect)
		self.pi = self.pi_prior * np.ones((self.num_classes,self.num_classes,self.num_annotators))

		# initialize estimated ground truth with majority voting
		self.ground_truth_est = np.zeros((self.n_train, self.num_classes))
		for i in range(self.n_train):
			votes = np.zeros(self.num_annotators)
			for r in range(self.num_annotators):
				if answers[i,r] != -1:
					votes[answers[i,r]] += 1
			self.ground_truth_est[i,np.argmax(votes)] = 1.0


	def e_step(self):
		print("E-step")
		for i in range(self.n_train):
			adjustment_factor = np.ones(self.num_classes)
			for r in range(self.num_annotators):
				if self.answers[i,r] != -1:
					adjustment_factor *= self.pi[:,self.answers[i,r],r]
			self.ground_truth_est[i,:] = np.transpose(adjustment_factor) * self.ground_truth_est[i,:]

		return self.ground_truth_est


	def m_step(self,):
		print("M-step")
		hist = self.model.fit(self.data_train, self.ground_truth_est, epochs=1, shuffle=True, batch_size=self.batch_size, verbose=0) 
		print(("loss:", hist.history["loss"][-1]))
		self.ground_truth_est = self.model.predict(self.data_train)

		self.pi = self.pi_prior * np.ones((self.num_classes,self.num_classes,self.num_annotators))
		for r in range(self.num_annotators):
			normalizer = np.zeros(self.num_classes)
			for i in range(self.n_train):
				if self.answers[i,r] != -1:
					self.pi[:,self.answers[i,r],r] += np.transpose(self.ground_truth_est[i,:])
					normalizer += self.ground_truth_est[i,:]
			normalizer = np.expand_dims(normalizer, axis=1)
			self.pi[:,:,r] = self.pi[:,:,r] / np.tile(normalizer, [1, self.num_classes])
		
		return self.model, self.pi


class CrowdsSequenceAggregator():

        def __init__(self, model, data_train, answers, batch_size = 16, pi_prior = 1.0):
                self.model = model
                self.data_train = data_train
                self.answers = answers
                self.batch_size = batch_size
                self.pi_prior = pi_prior
                self.n_train = answers.shape[0]
                self.seq_length = answers.shape[1]
                self.num_classes = np.max(answers) + 1
                self.num_annotators = answers.shape[2]
                print(("n_train:", self.n_train))
                print(("seq_length:", self.seq_length))
                print(("num_annotators:", self.num_annotators))

                # initialize annotators as reliable (almost perfect)
                self.pi = self.pi_prior * np.ones((self.num_classes,self.num_classes,self.num_annotators))

                # initialize estimated ground truth with majority voting
                self.ground_truth_est = np.zeros((self.n_train, self.seq_length, self.num_classes))
                for i in range(self.n_train):
                        for j in range(self.seq_length):
                                votes = np.zeros(self.num_annotators)
                                for r in range(self.num_annotators):
                                        if answers[i,j,r] != -1:
                                                votes[answers[i,j,r]] += 1
                                self.ground_truth_est[i,j,np.argmax(votes)] = 1.0


        def e_step(self):
                print("E-step")
                for i in range(self.n_train):
                        for j in range(self.seq_length):
                                adjustment_factor = np.ones(self.num_classes)
                                for r in range(self.num_annotators):
                                        if self.answers[i,j,r] != -1:
                                                adjustment_factor *= self.pi[:,self.answers[i,j,r],r]
                                self.ground_truth_est[i,j,:] = np.transpose(adjustment_factor) * self.ground_truth_est[i,j,:]

                return self.ground_truth_est


        def m_step(self,epochs):
                print("M-step")
                hist = self.model.fit(self.data_train, self.ground_truth_est, epochs=epochs, shuffle=True, batch_size=self.batch_size, verbose=0)
                print(("loss:", hist.history["loss"][-1]))
                self.ground_truth_est = self.model.predict(self.data_train)

                self.pi = self.pi_prior * np.ones((self.num_classes,self.num_classes,self.num_annotators))
                for r in range(self.num_annotators):
                        normalizer = np.zeros(self.num_classes)
                        for i in range(self.n_train):
                                for j in range(self.seq_length):
                                        if self.answers[i,j,r] != -1:
                                                self.pi[:,self.answers[i,j,r],r] += np.transpose(self.ground_truth_est[i,j,:])
                                                normalizer += self.ground_truth_est[i,j,:]
                        normalizer = np.expand_dims(normalizer, axis=1)
                        self.pi[:,:,r] = self.pi[:,:,r] / np.tile(normalizer, [1, self.num_classes])

                return self.model, self.pi

		
