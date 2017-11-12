
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import Layer

def init_identities(shape, dtype=None):
	out = np.zeros(shape)
	for r in xrange(shape[2]):
		for i in xrange(shape[0]):
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
			self.kernel = self.add_weight(shape=(self.output_dim, self.output_dim, self.num_annotators),
											initializer=init_identities, 
											trainable=True)
		elif self.conn_type == "VW":
			# vector of weights (one scale per class) per annotator
			self.kernel = self.add_weight(shape=(self.output_dim, self.num_annotators),
											initializer=keras.initializers.Ones(), 
											trainable=True)
		elif self.conn_type == "VW+B":
			# two vectors of weights (one scale and one bias per class) per annotator
			self.kernel = []
			self.kernel.append(self.add_weight(shape=(self.output_dim, self.num_annotators),
											initializer=keras.initializers.Ones(),
											trainable=True))
			self.kernel.append(self.add_weight(shape=(self.output_dim, self.num_annotators),
											initializer=keras.initializers.Zeros(),
											trainable=True))
		elif self.conn_type == "SW":
			# single weight value per annotator
			self.kernel = self.add_weight(shape=(self.num_annotators,1),
											initializer=keras.initializers.Ones(),
											trainable=True)
		else:
			raise Exception("Unknown connection type for CrowdsClassification layer!")

		super(CrowdsClassification, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		if self.conn_type == "MW":
			res = K.dot(x, self.kernel)
		elif self.conn_type == "VW" or self.conn_type == "VW+B" or self.conn_type == "SW":
			out = []
			for r in range(self.num_annotators):
				if self.conn_type == "VW":
					out.append(x * self.kernel[:,r])
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
			self.kernel.append(self.add_weight(shape=(1, self.num_annotators),
								  initializer=keras.initializers.Ones(),
								  trainable=True))
		elif self.conn_type == "B":
			# bias-only parameter
			self.kernel.append(self.add_weight(shape=(1, self.num_annotators),
								  initializer=keras.initializers.Zeros(),
								  trainable=True))
		elif self.conn_type == "S+B" or self.conn_type == "B+S":
			# scale and bias parameters
			self.kernel.append(self.add_weight(shape=(1, self.num_annotators),
									  initializer=keras.initializers.Ones(),
									  trainable=True))
			self.kernel.append(self.add_weight(shape=(1, self.num_annotators),
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


