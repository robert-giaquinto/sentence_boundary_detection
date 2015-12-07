from __future__ import division
import pickle
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sknn.mlp import Classifier, Convolution, Layer


# implement a standard random forest algorithm to evaluate current feature set

data_dir = '/Users/robert/Documents/UMN/5551_Robots/Project/data/'
df = pd.read_csv(data_dir + 'training.csv')
x = df[df.columns[3:]]
y = df[df.columns[2]]

# split intro test and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2015)

# Normalize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# create neural network with same architecture are my code
nn = Classifier(
	layers=[
		Layer("Tanh", units=50),
		Layer("Softmax")],
	learning_rate=0.01,
	batch_size=1000,
	valid_size=0.1,
	regularize="L2",
	n_iter=25,
	verbose=True)
nn.fit(x_train, y_train)
nn.score(x_train, y_train)
nn.score(x_test, y_test)
pickle.dump(nn, open('nn.pkl', 'wb'))
# nn = pickle.load(open('nn.pkl', 'rb'))


# try a deeper neural network with some better regularization
dnn = Classifier(
	layers=[
		Layer("Tanh", units=250),
		Layer("Maxout", units=100, pieces=2),
		Layer("Softmax")],
	learning_rate=0.01,
	batch_size=1000,
	valid_size=0.1,
	regularize="L2",
	dropout_rate=0.5,
	n_iter=25,
	verbose=True)
dnn.fit(x_train, y_train)
dnn.score(x_train, y_train)
dnn.score(x_test, y_test)
pickle.dump(dnn, open('dnn.pkl', 'wb'))
# dnn = pickle.load(open('dnn.pkl', 'rb'))


# Try a deep learning, convolutional neural net
cnn = Classifier(
	layers=[
		# Layer("Tanh", units=100),
		Convolution("Rectifier", channels=8, kernel_shape=(3, 3)),
		Layer("Softmax")],
	learning_rate=0.01,
	valid_size=0.1,
	n_iter=5,
	verbose=True)
cnn.fit(x_train, y_train)
cnn.score(x_train, y_train)
cnn.score(x_test, y_test)
pickle.dump(cnn, open('cnn.pkl', 'wb'))
# cnn = pickle.load(open('cnn.pkl', 'rb'))
