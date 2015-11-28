from __future__ import division
import numpy as np
import logging
import math
from Layers import Input_Layer, Output_Layer, Hidden_Layer
from helper_funcs import *


# TOOD: create a validation set based method of early stopping
class MLP(object):
	def __init__(self, num_hidden_layers, alpha=.1, lambda_penalty=0.0001, threshold=0.04,
			activation="sigmoid", random_seed=None):
		if type(num_hidden_layers) == int:
			self.num_hidden_layers = [num_hidden_layers]
		elif type(num_hidden_layers) == list:
			self.num_hidden_layers = num_hidden_layers
		else:
			print "num_hidden_layers must be either an int or list of ints"
		self.alpha = alpha
		self.lambda_penalty = lambda_penalty
		self.threshold = threshold # THIS IS BAD FIX THIS APPROACH
		self.activation = activation
		self.random_seed = random_seed
		self.layers = []

	def _initialize_layers(self, architecture):
		if self.random_seed is not None:
			random_state = np.random.RandomState(self.random_seed)
		else:
			random_state = np.random.RandomState(123)

		# pass the loss and error functions to use
		compute_error = lambda yhat, y: np.multiply(y - yhat, d_activation(y))
		compute_loss = lambda yhat, y: 0.5 * np.sum((yhat - y) ** 2)

		if self.activation == "sigmoid":
			activation = sigmoid
			d_activation = d_sigmoid
		elif self.activation == "tanh":
			activation = np.tanh
			d_activation = d_tanh

		num_layers = len(architecture)
		for i in range(num_layers)[::-1]:
			if i == num_layers - 1:
				# output layer
				self.layers.append(Output_Layer(architecture[i], activation, d_activation, compute_error=compute_error, compute_loss=compute_loss))
			elif i == 0:
				# input layer
				self.layers.append(Input_Layer(architecture[i], architecture[i + 1], activation, d_activation, self.layers[-1], alpha=self.alpha, lambda_penalty=self.lambda_penalty, random_state=random_state))
			else:
				# hidden layer(s)
				self.layers.append(Hidden_Layer(architecture[i], architecture[i + 1], activation, d_activation, self.layers[-1], alpha=self.alpha, lambda_penalty=self.lambda_penalty, random_state=random_state))

	def fit(self, x, y, epochs, batch_size=None):
		num_obs = x.shape[1]
		# initialize network architecture
		architecture = [x.shape[0]] + self.num_hidden_layers + [y.shape[0]]
		self._initialize_layers(architecture)

		# determine batch size if SGD requested
		if batch_size is None:
			# simply define batch size to be the whole set
			batch_size = num_obs
		# how many batches are needed?
		num_batches = max(1, int(math.ceil(num_obs / batch_size)))

		# begin training
		previous_loss = np.inf
		no_dif_counter = 0
		for i in range(epochs):
			# if SGD requested, select a random subset of points to train on
			for batch in range(num_batches):
				start = batch * batch_size
				end = min(num_obs, (batch + 1) * batch_size)
				loss = self._fit_partial(x[:, start:end], y[:, start:end])
				logging.info("Epoch %d: Cost = %f" % (i, loss))
			loss_difference = abs(loss - previous_loss)
			previous_loss = loss
			if loss_difference < 0.0001 and loss < self.threshold:
				no_dif_counter += 1
			else:
				no_dif_counter = 0
			if no_dif_counter > math.floor(epochs/10):
				print "STOP EARLY"
				logging.info("Epoch %d: Cost = %f   %d" % (i, loss, no_dif_counter))
				break

	def _fit_partial(self, x, y):
		input_layer = self.layers[-1]
		output_layer = self.layers[0]
		output_layer.set_y(y)
		input_layer.feed_forward(x, backprop=True)
		loss = sum([layer.loss for layer in self.layers])
		return loss

	def predict_proba(self, x):
		input_layer = self.layers[-1]
		input_layer.feed_forward(x, backprop=False)
		output_layer = self.layers[0]
		probs = output_layer.get_yhat()
		return probs

	def predict(self, x):
		probs = self.predict_proba(x)
		yhat = np.argmax(probs, axis=0)
		return yhat

	def score_classes(self, x, y):
		yhat = self.predict(x)
		y_flat = np.argmax(y, axis=0)
		counts = [0] * y.shape[0]
		corrects = [0] * y.shape[0]
		for i in range(len(yhat)):
			klass = y_flat[i]
			if klass == yhat[i]:
				corrects[klass] += 1
			counts[klass] += 1
		rval = [1.0 * corr / ct for corr, ct in zip(corrects, counts)]
		return rval

	def score(self, x, y):
		yhat = self.predict(x)
		y_flat = np.argmax(y, axis=0)
		return 1.0 * sum(y_flat == yhat) / len(y_flat)