from __future__ import division
import numpy as np


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
	return np.ones(x.shape) - (x ** 2)

def squared_error(y, est):
	num_classes = y.shape[1]
	rval = 0
	for c in range(num_classes):
		est_c = est[np.where(y[:,c] == 1)[0]]
		num_class_c = np.sum(y[:,c])
		rval += np.sum(np.pow(np.subtract(np.ones([num_class_c, 1]), est_c), 2))
	return rval
