from __future__ import division
import numpy as np
import random
import math


def split_validation(y, pct_validation, random_seed):
	"""
	Statified sampling of indices to create equal representation of y in each split
	:param y:
	:param pct_validation:
	:return:
	"""
	num_classes = y.shape[0]
	y_flat = np.argmax(y, axis=0)
	num_obs = y.shape[1]
	in_train = []
	for c in range(num_classes):
		in_this_class = np.where(y_flat == c)[0]
		num_selected = int(math.ceil((1.0 - pct_validation) * len(in_this_class)))
		in_train += random.sample(in_this_class, num_selected)
	in_validation = list(set(range(num_obs)) - set(in_train))
	rval = {'train': in_train, 'validation': in_validation}
	return rval


def k_fold(y, k):
	"""
	given the target variable y (a numpy array),
	and number of folds k (int),
	this returns a list of length k sublists each containing the
	row numbers of the items in the training set
	NOTE: THIS IS STRATAFIED K-FOLD CV (i.e. classes remained balanced)
	"""
	targets = np.unique(y)
	rval = []
	for fold in range(k):
		in_train = []
		for tar in targets:
			# how many can be select from?
			num_in_this_class = len(y[y == tar])

			# how many will be selected
			num_in_training = int(round(num_in_this_class * (k-1)/k))

			# indices of those who can be selected
			in_this_class = np.where(y == tar)[0]

			# add selected indices to the list of training samples
			in_train += random.sample(in_this_class, num_in_training)
		rval.append(np.array(in_train))
	return np.array(rval)

def make_binary(x):
	# convert the class labels to a binary matrix
	num_obs = x.shape[0]
	classes = sorted(np.unique(x))
	rval = np.zeros([num_obs, len(classes)])
	for i in range(num_obs):
		col = np.where(x[i] == classes)[0][0]
		rval[i, col] = 1
	return rval


def sigmoid(x):
	# vectorized version of the sigmoid function
	return 1.0 / (1 + np.exp(-1.0 * x))


def d_sigmoid(x):
	# derivative of sigmoid function
	return x * (np.ones(x.shape) - x)


def d_tanh(x):
	# derivative of tanh activation function
	return np.ones(x.shape) - np.power(x, 2)

def squared_error(y, est):
	num_classes = y.shape[1]
	rval = 0
	for c in range(num_classes):
		est_c = est[np.where(y[:,c] == 1)[0]]
		num_class_c = np.sum(y[:,c])
		rval += np.sum(np.pow(np.subtract(np.ones([num_class_c, 1]), est_c), 2))
	return rval
