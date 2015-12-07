from __future__ import division
import numpy as np
from helper_funcs import *


class Base_Layer(object):
	def __init__(self, activation, d_activation):
		self.activation = activation
		self.d_activation = d_activation

	def _backpropagate(self, x):
		pass

	def feed_forward(self, x):
		pass


class Input_Layer(object):
	def __init__(self, num_inputs, num_outputs, activation, d_activation, next_layer,
			alpha=0.1, lambda_penalty=0.0001, random_state=None):
		# initialize weights and biases based on deep learning book recommendation
		self.W = 4 * np.asarray(random_state.uniform(
				low=-np.sqrt(6.0 / (num_inputs + num_outputs)),
				high=np.sqrt(6.0 / (num_inputs + num_outputs)),
				size=(num_outputs, num_inputs)), dtype=np.float64)
		self.b = np.zeros(num_outputs, dtype=np.float64).reshape((num_outputs, 1))
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.activation = activation
		self.d_activation = d_activation
		self.next_layer = next_layer
		self.alpha = alpha
		self.lambda_penalty = lambda_penalty

	def _backpropagate(self, x):
		delta_hidden = self.next_layer.get_delta()
		num_obs = x.shape[1]
		self.W -= self.alpha * ((np.dot(delta_hidden, x.T) / num_obs) + self.lambda_penalty * self.W)
		self.b -= self.alpha * (np.sum(delta_hidden, axis=1).reshape(self.num_outputs, 1) / num_obs)

	def feed_forward(self, x, backprop):
		num_obs = x.shape[1]
		z1 = np.dot(self.W, x) + np.tile(self.b, (1, num_obs))
		a1 = self.activation(z1)
		self.loss = self.lambda_penalty / 2 * (self.W ** 2).sum()
		self.next_layer.feed_forward(a1, backprop)
		if backprop:
			self._backpropagate(x)


class Hidden_Layer(object):
	"""
	Hidden layer of multilayer perceptron.
	Implements a softmax activation function for classification, however
	paramter activation refers to the activation function used between
	input and hidden layers.
	"""
	def __init__(self, num_inputs, num_outputs, activation, d_activation, next_layer,
			alpha=0.1, lambda_penalty=0.0001, random_state=None):
		# initialize weights and biases based on deep learning book recommendation
		self.W = np.asarray(random_state.uniform(
				low=-np.sqrt(6.0 / (num_inputs + num_outputs)),
				high=np.sqrt(6.0 / (num_inputs + num_outputs)),
				size=(num_outputs, num_inputs)), dtype=np.float64) * 4.0
		self.b = np.zeros(num_outputs, dtype=np.float64).reshape((num_outputs, 1))
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.activation = activation
		self.d_activation = d_activation
		self.next_layer = next_layer
		self.alpha = alpha
		self.lambda_penalty = lambda_penalty
		self.loss = None

	def get_delta(self):
		"""
		pass delta for gradient descent to input/hidden layer
		:return:
		"""
		return self.delta

	def _backpropagate(self, a1):
		num_obs = a1.shape[1]
		delta = self.next_layer.get_delta()
		self.W -= self.alpha * ((np.dot(delta, a1.T) / num_obs) + self.lambda_penalty * self.W)
		self.b -= self.alpha * (np.sum(delta, axis=1).reshape((self.num_outputs, 1)) / num_obs)
		self.delta = np.dot(self.W.T, delta) * self.d_activation(a1)

	def feed_forward(self, a1, backprop):
		num_obs = a1.shape[1]
		z2 = np.dot(self.W, a1) + np.tile(self.b, (1, num_obs))
		exp_z = np.exp(z2)
		a2 = exp_z / np.sum(exp_z, axis=0, keepdims=True)
		self.loss = self.lambda_penalty / 2 * (self.W ** 2).sum()
		self.next_layer.feed_forward(a2, backprop)
		if backprop:
			self._backpropagate(a1)


class Output_Layer(object):
	def __init__(self, num_inputs):
		self.num_inputs = num_inputs

	def set_y(self, y):
		"""
		initialize target values at output layer.
		:param y:
		:return:
		"""
		self.y = y

	def get_delta(self):
		"""
		pass delta for gradient descent to hidden layers.
		:return:
		"""
		return self.delta

	def get_yhat(self):
		return self.yhat

	def _backpropagate(self, a2):
		num_obs = a2.shape[1]
		self.loss = -1.0 * (np.sum(np.multiply(self.y, np.log(a2)))) / num_obs
		# softmax:
		a2[self.y[1, :], range(num_obs)] -= 1
		self.delta = a2

	def feed_forward(self, a2, backprop):
		self.yhat = a2
		if backprop:
			self._backpropagate(a2)
