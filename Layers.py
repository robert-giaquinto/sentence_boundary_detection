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
	def __init__(self, n_in, n_out, activation, d_activation, next_layer, alpha=0.1, lambda_penalty=0.0001, random_state=None):
		self.W = 4 * np.asarray(random_state.uniform(
				low=-np.sqrt(6.0 / (n_in + n_out)),
				high=np.sqrt(6.0 / (n_in + n_out)),
				size=(n_out, n_in)), dtype=np.float64)
		self.b = np.zeros(n_out, dtype=np.float64).reshape((n_out, 1))
		self.n_in = n_in
		self.n_out = n_out
		self.activation = activation
		self.d_activation = d_activation
		self.next_layer = next_layer
		self.alpha = alpha
		self.lambda_penalty = lambda_penalty

	def feed_forward(self, x, backprop):
		num_obs = x.shape[1]
		z = np.dot(self.W, x) + np.tile(self.b, (1, num_obs))
		a_out = self.activation(z)
		self.loss = self.lambda_penalty * (self.W ** 2).sum()
		self.next_layer.feed_forward(a_out, backprop)
		if backprop:
			self._backpropagate(x)

	def _backpropagate(self, x):
		delta = self.next_layer.get_delta()
		num_obs = x.shape[1]
		self.W -= self.alpha * np.dot(delta, x.T) / num_obs
		self.b -= self.alpha * np.sum(delta, axis=1).reshape(self.n_out, 1) / num_obs


class Hidden_Layer(object):
	def __init__(self, n_in, n_out, activation, d_activation, next_layer, alpha=0.1, lambda_penalty=0.0001, random_state=None):
		self.W = np.asarray(random_state.uniform(
				low=-np.sqrt(6.0 / (n_in + n_out)),
				high=np.sqrt(6.0 / (n_in + n_out)),
				size=(n_out, n_in)), dtype=np.float64) * 4.0
		self.b = np.zeros(n_out, dtype=np.float64).reshape((n_out, 1))
		self.n_in = n_in
		self.n_out = n_out
		self.activation = activation
		self.d_activation = d_activation
		self.next_layer = next_layer
		self.alpha = alpha
		self.lambda_penalty = lambda_penalty

	def get_delta(self):
		"""
		pass delta for gradient descent to input/hidden layer
		:return:
		"""
		return self.delta

	def _backpropagate(self, a_in):
		num_obs = a_in.shape[1]
		delta = self.next_layer.get_delta()
		self.W -= self.alpha * np.dot(delta, a_in.T) / num_obs
		self.b -= self.alpha * np.sum(delta, axis=1).reshape((self.n_out, 1)) / num_obs
		self.delta = np.dot(self.W.T, delta) * self.d_activation(a_in)

	def feed_forward(self, a_in, backprop):
		num_obs = a_in.shape[1]
		z = np.dot(self.W, a_in) + np.tile(self.b, (1, num_obs))
		a_out = self.activation(z)
		self.loss = self.lambda_penalty * (self.W ** 2).sum()
		self.next_layer.feed_forward(a_out, backprop)
		if backprop:
			self._backpropagate(a_in)


class Output_Layer(object):
	def __init__(self, n_in, activation, d_activation, compute_error, compute_loss):
		self.n_in = n_in
		self.activation = activation
		self.d_activation = d_activation
		self.compute_error = compute_error
		self.compute_loss = compute_loss

	def set_y(self, y):
		"""
		initialize target values at output layer
		:param y:
		:return:
		"""
		self.y = y

	def get_delta(self):
		"""
		pass delta for gradient descent to hidden layers
		:return:
		"""
		return self.delta

	def get_yhat(self):
		return self.yhat

	def _backpropagate(self, a_in):
		num_obs = a_in.shape[1]
		self.delta = self.compute_error(self.y, a_in)
		self.loss = self.compute_loss(self.y, a_in) / num_obs

	def feed_forward(self, a_in, backprop):
		self.yhat = a_in
		if backprop:
			self._backpropagate(a_in)
