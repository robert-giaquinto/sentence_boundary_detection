from __future__ import division
import numpy as np
import logging
import math
from Layers import Input_Layer, Output_Layer, Hidden_Layer
from helper_funcs import *


class MLP(object):
	def __init__(self, num_hidden_units, alpha=.1, lambda_penalty=0.0001, activation="sigmoid", random_seed=None):
		if type(num_hidden_units) == int:
			self.num_hidden_units = [num_hidden_units]
		elif type(num_hidden_units) == list:
			self.num_hidden_units = num_hidden_units
		else:
			print "num_hidden_units must be either an int or list of ints"
		self.alpha = alpha
		self.lambda_penalty = lambda_penalty
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

	def fit(self, x, y, epochs, batch_size=None, pct_validation=.25):
		# initialize network architecture
		architecture = [x.shape[0]] + self.num_hidden_units + [y.shape[0]]
		self._initialize_layers(architecture)

		# split out a validation set if requested
		if pct_validation <= 0 or pct_validation >= 1:
			print "pct_validation must be between (0,1), resorting to default"
			pct_validation = .25

		splits = split_validation(y, pct_validation)
		x_train = x[:, splits['train']]
		y_train = y[:, splits['train']]
		x_valid = x[:, splits['validation']]
		y_valid = y[:, splits['validation']]

		# determine number of batches to run for each epoch (if SGD requested)
		num_obs = x_train.shape[1]
		if batch_size is None:
			# simply define batch size to be the whole set
			batch_size = num_obs
		# how many batches are needed?
		num_batches = max(1, int(math.ceil(num_obs / batch_size)))

		# begin training
		previous_score = -np.inf
		stall_count = 0
		stall_limit = int(math.floor(epochs/10))
		for i in range(epochs):
			# if SGD requested, select a random subset of points to train on
			for batch in range(num_batches):
				start = batch * batch_size
				end = min(num_obs, (batch + 1) * batch_size)
				loss = self._fit_partial(x_train[:, start:end], y_train[:, start:end])
			train_score = self.score(x_train, y_train)
			valid_score = self.score(x_valid, y_valid)
			logging.info("Epoch %d:" % i)
			logging.info("Loss = %f" % loss)
			logging.info("Train Accuracy = %f" % train_score)
			logging.info("Validation Accuracy = %f\n" % valid_score)

			score_dif = valid_score - previous_score
			if score_dif < 0.01:
				stall_count += 1
			else:
				stall_count = 0
				previous_score = valid_score
			if stall_count > stall_limit:
				logging.info("STOP EARLY, no improvement in %s epochs" % stall_limit)
				logging.info("Epoch %d:" % i)
				logging.info("Loss = %f" % loss)
				logging.info("Train Accuracy = %f" % train_score)
				logging.info("Validation Accuracy = %f\n" % valid_score)
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