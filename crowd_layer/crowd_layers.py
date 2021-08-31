
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
#from tensorflow.keras.engine.topology import Layer

def init_identities(shape, dtype=None):
	out = np.zeros(shape)
	for r in range(shape[2]):
		for i in range(shape[0]):
			out[i,i,r] = 1.0
	return out
	
class CrowdsClassification(Layer):

	def __init__(self, output_dim, num_annotators, conn_type="MW", **kwargs):
		self.output_dim = output_dim
		self.num_annotators = num_annotators
		self.conn_type = conn_type
		super(CrowdsClassification, self).__init__(**kwargs)

	def build(self, input_shape):
		if self.conn_type == "MW":
			# matrix of weights per annotator
			self.kernel = self.add_weight("CrowdLayer", (self.output_dim, self.output_dim, self.num_annotators),
											initializer=init_identities, 
											trainable=True)
		elif self.conn_type == "VW":
			# vector of weights (one scale per class) per annotator
			self.kernel = self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
											initializer=keras.initializers.Ones(), 
											trainable=True)
		elif self.conn_type == "VB":
			# two vectors of weights (one scale and one bias per class) per annotator
			self.kernel = []
			self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
											initializer=keras.initializers.Zeros(),
											trainable=True))
		elif self.conn_type == "VW+B":
			# two vectors of weights (one scale and one bias per class) per annotator
			self.kernel = []
			self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
											initializer=keras.initializers.Ones(),
											trainable=True))
			self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
											initializer=keras.initializers.Zeros(),
											trainable=True))
		elif self.conn_type == "SW":
			# single weight value per annotator
			self.kernel = self.add_weight("CrowdLayer", (self.num_annotators,1),
											initializer=keras.initializers.Ones(),
											trainable=True)
		else:
			raise Exception("Unknown connection type for CrowdsClassification layer!")

		super(CrowdsClassification, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		if self.conn_type == "MW":
			res = K.dot(x, self.kernel)
		elif self.conn_type == "VW" or self.conn_type == "VB" or self.conn_type == "VW+B" or self.conn_type == "SW":
			out = []
			for r in range(self.num_annotators):
				if self.conn_type == "VW":
					out.append(x * self.kernel[:,r])
				elif self.conn_type == "VB":
					out.append(x + self.kernel[0][:,r])
				elif self.conn_type == "VW+B":
					out.append(x * self.kernel[0][:,r] + self.kernel[1][:,r])
				elif self.conn_type == "SW":
					out.append(x * self.kernel[r,0])
			res = tf.stack(out)
			if len(res.shape) == 3:
				res = tf.transpose(res, [1, 2, 0])
			elif len(res.shape) == 4:
				res = tf.transpose(res, [1, 2, 3, 0])
			else:
				raise Exception("Wrong number of dimensions for output")
		else:
			raise Exception("Unknown connection type for CrowdsClassification layer!") 
		
		return res

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim, self.num_annotators)


class CrowdsRegression(Layer):

	def __init__(self, num_annotators, conn_type="B", **kwargs):
		self.num_annotators = num_annotators
		self.conn_type = conn_type
		super(CrowdsRegression, self).__init__(**kwargs)

	def build(self, input_shape):
		self.kernel = []
		if self.conn_type == "S":
			# scale-only parameter
			self.kernel.append(self.add_weight("CrowdLayer", (1, self.num_annotators),
								  initializer=keras.initializers.Ones(),
								  trainable=True))
		elif self.conn_type == "B":
			# bias-only parameter
			self.kernel.append(self.add_weight("CrowdLayer", (1, self.num_annotators),
								  initializer=keras.initializers.Zeros(),
								  trainable=True))
		elif self.conn_type == "S+B" or self.conn_type == "B+S":
			# scale and bias parameters
			self.kernel.append(self.add_weight("CrowdLayer", (1, self.num_annotators),
									  initializer=keras.initializers.Ones(),
									  trainable=True))
			self.kernel.append(self.add_weight("CrowdLayer", (1, self.num_annotators),
									  initializer=keras.initializers.Zeros(),
									  trainable=True))
		else:
			raise Exception("Unknown connection type for CrowdsRegression layer!") 

		super(CrowdsRegression, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		if self.conn_type == "S":
			#res = K.dot(x, self.kernel[0])
			res = x * self.kernel[0]
		elif self.conn_type == "B":
			res = x + self.kernel[0]
		elif self.conn_type == "S+B":
			#res = K.dot(x, self.kernel[0]) + self.kernel[1]
			res = x * self.kernel[0] + self.kernel[1]
		elif self.conn_type == "B+S":
			res = (x + self.kernel[1]) * self.kernel[0]
		else:
			raise Exception("Unknown connection type for CrowdsClassification layer!") 

		return res

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.num_annotators)


class MaskedMultiCrossEntropy(object):

	def loss(self, y_true, y_pred):
		vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=1)
		mask = tf.equal(y_true[:,0,:], -1)
		zer = tf.zeros_like(vec)
		loss = tf.where(mask, x=zer, y=vec)
		return loss


class MaskedMultiMSE(object):
		
	def loss(self, y_true, y_pred):
		vec = K.square(y_pred - y_true)
		mask = tf.equal(y_true[:,:], 999999999)
		zer = tf.zeros_like(vec)
		loss = tf.where(mask, x=zer, y=vec)
		return loss


class MaskedMultiSequenceCrossEntropy(object):

	def __init__(self, num_classes):
		self.num_classes = num_classes

	def loss(self, y_true, y_pred):
		mask_missings = tf.equal(y_true, -1)
		mask_padding = tf.equal(y_true, 0)

		# convert targets to one-hot enconding and transpose
		y_true = tf.transpose(tf.one_hot(tf.cast(y_true, tf.int32), self.num_classes, axis=-1), [0,1,3,2])

		# masked cross-entropy
		vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=2)
		zer = tf.zeros_like(vec)
		vec = tf.where(mask_missings, x=zer, y=vec)
		vec = tf.where(mask_padding, x=zer, y=vec)
		loss = tf.reduce_mean(vec, axis=-1)
		return loss


class CrowdsAggregationCategoricalCrossEntropy(object):

	def __init__(self, num_classes, num_annotators, pi_prior=0.01):
		self.num_classes = num_classes
		self.num_annotators = num_annotators
		self.pi_prior = pi_prior
		
		# initialize pi_est (annotators' estimated confusion matrices) wit identities
		self.pi_est = np.zeros((self.num_classes,self.num_classes,self.num_annotators), dtype=np.float32)
		for r in range(self.num_annotators):
			self.pi_est[:,:,r] = np.eye(self.num_classes) + self.pi_prior
			self.pi_est[:,:,r] /= np.sum(self.pi_est[:,:,r], axis=1)
			
		self.init_suff_stats()
			
	def init_suff_stats(self):
		# initialize suff stats for M-step
		self.suff_stats = self.pi_prior * tf.ones((self.num_annotators,self.num_classes,self.num_classes))
		
	def loss_fc(self, y_true, y_pred):
		y_true = tf.cast(y_true, tf.int32)

		#y_pred += 0.01
		#y_pred /= tf.reduce_sum(y_pred, reduction_indices=len(y_pred.get_shape()) - 1, keep_dims=True)

		#y_pred = tf.where(tf.less(y_pred, 0.001), 
		#                        #0.01 * tf.ones_like(y_pred), 
		#                        0.001 + y_pred, 
		#                        y_pred)
		#y_pred += 0.01 # y_pred cannot be zero!
		eps = 1e-3
		#y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
		y_pred = tf.clip_by_value(y_pred, eps, 9999999999)


		# E-step
		adjustment_factor = tf.ones_like(y_pred)
		for r in range(self.num_annotators):
			adj = tf.where(tf.equal(y_true[:,r], -1), 
								tf.ones_like(y_pred), 
								tf.gather(tf.transpose(self.pi_est[:,:,r]), y_true[:,r]))
			adjustment_factor = tf.multiply(adjustment_factor, adj)
			
		res = tf.multiply(adjustment_factor, y_pred)
		y_agg = res / tf.expand_dims(tf.reduce_sum(res, axis=1), 1)

		loss = -tf.reduce_sum(y_agg * tf.log(y_pred), reduction_indices=[1])
		
		# update suff stats
		upd_suff_stats = []
		for r in range(self.num_annotators):
			#print r
			suff_stats = []
			normalizer = tf.zeros_like(y_pred)
			for c in range(self.num_classes):
				suff_stats.append(tf.reduce_sum(tf.where(tf.equal(y_true[:,r], c), 
									y_agg,
									tf.zeros_like(y_pred)), axis=0))
			upd_suff_stats.append(suff_stats)
		upd_suff_stats = tf.stack(upd_suff_stats)
		self.suff_stats += upd_suff_stats

		return loss
	
	def m_step(self):
		#print "M-step"
		self.pi_est = tf.transpose(self.suff_stats / tf.expand_dims(tf.reduce_sum(self.suff_stats, axis=2), 2), [1, 2, 0])
		
		return self.pi_est


class CrowdsAggregationBinaryCrossEntropy(object):

	def __init__(self, num_annotators, pi_prior=0.01, alpha=None, beta=None, update_freq=1):
		self.num_annotators = num_annotators
		self.pi_prior = pi_prior
		self.alpha = alpha
		self.beta = beta
		self.update_freq = update_freq
		
		# initialize alpha and beta (annotators' estimated sensitivity and specificity)
		if self.alpha == None:
			print("initializing alpha with unit...")
			self.alpha = 0.99*np.ones((self.num_annotators,1), dtype=np.float32)
		if self.beta == None:
			self.beta = 0.99*np.ones((self.num_annotators,1), dtype=np.float32)
		self.count = tf.ones(1)
			
		self.suff_stats_alpha = [self.pi_prior for r in range(self.num_annotators)]
		self.suff_stats_beta = [self.pi_prior for r in range(self.num_annotators)]
		self.suff_stats_alpha_norm = [self.pi_prior for r in range(self.num_annotators)]
		self.suff_stats_beta_norm = [self.pi_prior for r in range(self.num_annotators)]
			
	def init_suff_stats(self):
		# initialize suff stats for M-step
		pass

	def loss_fc(self, y_true, y_pred):
		#y_true = tf.cast(y_true, tf.int32)

		#y_pred += 0.01
		#y_pred /= tf.reduce_sum(y_pred, reduction_indices=len(y_pred.get_shape()) - 1, keep_dims=True)

		#y_pred = tf.where(tf.less(y_pred, 0.001), 
		#                        #0.01 * tf.ones_like(y_pred), 
		#                        0.001 + y_pred, 
		#                        y_pred)
		#y_pred += 0.01 # y_pred cannot be zero!
		eps = 1e-3
		y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
		#y_pred = tf.clip_by_value(y_pred, eps, 9999999999)

		p = y_pred[:,1]
		self.count += 1
		self.count = tf.Print(self.count, [self.count])
		#self.count += 1

		if False:
			print("M-step...")
			self.alpha = []
			self.beta = []
			for r in range(self.num_annotators):
				self.alpha.append(self.suff_stats_alpha[r] / self.suff_stats_alpha_norm[r])
				self.beta.append(self.suff_stats_beta[r] / self.suff_stats_beta_norm[r])
			self.count = 0
			self.suff_stats_alpha = [self.pi_prior for r in range(self.num_annotators)]
			self.suff_stats_beta = [self.pi_prior for r in range(self.num_annotators)]
			self.suff_stats_alpha_norm = [self.pi_prior for r in range(self.num_annotators)]
			self.suff_stats_beta_norm = [self.pi_prior for r in range(self.num_annotators)]
			self.alpha = tf.Print(self.alpha, [self.alpha])

		
		# E-step
		a = tf.ones_like(p)
		b = tf.ones_like(p)
		for r in range(self.num_annotators):
			a = a * tf.where(tf.equal(y_true[:,r], 1), self.alpha[r]*tf.ones_like(p), tf.ones_like(p))
			b = b * tf.where(tf.equal(y_true[:,r], 1), (1.0-self.beta[r])*tf.ones_like(p), tf.ones_like(p))
			a = a * tf.where(tf.equal(y_true[:,r], 0), (1.0-self.alpha[r])*tf.ones_like(p), tf.ones_like(p))
			b = b * tf.where(tf.equal(y_true[:,r], 0), self.beta[r]*tf.ones_like(p), tf.ones_like(p))
		
		mu = (a*p) / (a*p + b*(1.0-p))
		#mu = tf.Print(mu, [mu])
		loss = - (mu * tf.log(y_pred[:,1]) + (1.0-mu) * tf.log(y_pred[:,0]))

		# update suff stats
		for r in range(self.num_annotators):
			self.suff_stats_alpha[r] += tf.reduce_sum(tf.where(tf.equal(y_true[:,r], 1), mu, tf.zeros_like(p)))
			self.suff_stats_beta[r] += tf.reduce_sum(tf.where(tf.equal(y_true[:,r], 0), (1.0-mu), tf.zeros_like(p)))
			self.suff_stats_alpha_norm[r] += tf.reduce_sum(tf.where(tf.equal(y_true[:,r], -1), tf.zeros_like(p), mu))
			self.suff_stats_beta_norm[r] += tf.reduce_sum(tf.where(tf.equal(y_true[:,r], -1), tf.zeros_like(p), (1.0-mu)))
		
		return loss
	
	def m_step(self):
		print((dir(self)))
		print(("debug:", self.count.eval()))
		#print "M-step"
		#self.count += 1
		#print "increment", self.count
		#if self.count >= self.update_freq:
			
			
		
		return (self.alpha, self.beta)


class CrowdsAggregationCallback(keras.callbacks.Callback):

	def __init__(self, loss):
		self.loss = loss
		
	def on_epoch_begin(self, epoch, logs=None):
		self.loss.init_suff_stats()
		
	def on_epoch_end(self, epoch, logs=None):
		# run M-step
		self.model.pi = self.loss.m_step()


