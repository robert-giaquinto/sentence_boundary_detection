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

		# define the activation function used between input and hidden layers
		if self.activation == "tanh":
			activation_func = np.tanh
			d_activation_func = d_tanh
		elif self.activation == "sigmoid":
			activation_func = sigmoid
			d_activation_func = d_sigmoid

		num_layers = len(architecture)
		for i in range(num_layers)[::-1]:
			if i == num_layers - 1:
				# output layer
				self.layers.append(Output_Layer(num_inputs=architecture[i]))
			elif i == 0:
				# input layer
				self.layers.append(Input_Layer(num_inputs=architecture[i], num_outputs=architecture[i + 1],
					activation=activation_func, d_activation=d_activation_func,
					next_layer=self.layers[-1], alpha=self.alpha, lambda_penalty=self.lambda_penalty,
					random_state=random_state))
			else:
				# hidden layer(s)
				self.layers.append(Hidden_Layer(num_inputs=architecture[i], num_outputs=architecture[i + 1],
					activation=activation_func, d_activation=d_activation_func,
					next_layer=self.layers[-1], alpha=self.alpha, lambda_penalty=self.lambda_penalty,
					random_state=random_state))

	def fit(self, x, y, epochs, batch_size, stall_limit=None, pct_validation=0.1):
		# initialize network architecture
		architecture = [x.shape[0]] + self.num_hidden_units + [y.shape[0]]
		self._initialize_layers(architecture)

		# split out a validation set if requested
		if pct_validation < 0 or pct_validation >= 1:
			print "pct_validation must be between (0,1), resorting to default"
			return 0
		elif pct_validation == 0:
			x_train = x
			y_train = y
		else:
			splits = split_validation(y, pct_validation)
			x_train = x[:, splits['train']]
			y_train = y[:, splits['train']]
			x_valid = x[:, splits['validation']]
			y_valid = y[:, splits['validation']]

		# determine number of batches to run for each epoch (if mini-batch requested)
		num_obs = x_train.shape[1]
		if batch_size is None:
			# simply define batch size to be the whole set
			batch_size = num_obs
		# how many batches are needed?
		num_batches = max(1, int(math.ceil(num_obs / batch_size)))

		# begin training
		previous_score = -np.inf
		stall_count = 0
		stall_limit = int(math.floor(epochs/10)) if stall_limit is None else stall_limit
		for i in range(epochs):
			# if mini-batch requested, select a subset of points to train on
			for batch in range(num_batches):
				start = batch * batch_size
				end = min(num_obs, (batch + 1) * batch_size)
				loss = self._fit_partial(x_train[:, start:end], y_train[:, start:end])
			train_score = self.score(x_train, y_train)
			valid_score = self.score(x_valid, y_valid)
			logging.info("\nEpoch %d:" % i)
			logging.info("Loss = %f" % loss)
			logging.info("Train Accuracy = %f" % train_score)
			if pct_validation > 0:
				logging.info("Valid Accuracy = %f" % valid_score)
				score_dif = valid_score - previous_score
			else:
				score_dif = train_score - previous_score

			# determine if performance is stalling out
			if score_dif < 0.01:
				stall_count += 1
			else:
				stall_count = 0
				if pct_validation > 0:
					previous_score = valid_score
				else:
					previous_score = train_score
			if stall_count > stall_limit:
				logging.info("\nSTOP EARLY, no improvement in %s epochs" % stall_limit)
				logging.info("Epoch %d:" % i)
				logging.info("Loss = %f" % loss)
				logging.info("Train Accuracy = %f" % train_score)
				if pct_validation > 0:
					logging.info("Valid Accuracy = %f" % valid_score)
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