from __future__ import division
import pickle
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sknn.mlp import Classifier, Layer
from sklearn.decomposition import PCA

data_dir = '/Users/robert/Documents/UMN/5551_Robots/Project/data/'
df = pd.read_csv(data_dir + 'training.csv')
x = df[df.columns[3:]]
y = df[df.columns[2]]

# split intro test and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2015)

# Normalize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# pca transformation on review variables
pca = PCA(n_components=30, whiten=True)
num_real_vars = 87
x_train_pca = pca.fit_transform(x_train[:, num_real_vars:])
x_train = np.hstack((x_train[:, 0:num_real_vars], x_train_pca))

x_test_pca = pca.transform(x_test[:, num_real_vars:])
x_test = np.hstack((x_test[:, 0:num_real_vars], x_test_pca))



# create neural network with same architecture are my code
batch_sizes = [50, 100]
learning_rates = [.1, .2]
num_epochs = np.arange(1, 16)
nn_accuracy = []
nn_batch_sizes = []
nn_alphas = []
nn_num_units = []
nn_dropout = []
nn_epochs = []
nn_type = []
for alpha in learning_rates:
	for batch_size in batch_sizes:
		for epoch in num_epochs:
			nn = Classifier(
				layers=[
					Layer("Tanh", units=100),
					Layer("Softmax")],
				learning_rate=alpha,
				batch_size=batch_size,
				regularize="L2",
				n_iter=epoch,
				verbose=False)
			nn.fit(x_train, y_train)
			test_score = nn.score(x_test, y_test)
			nn_accuracy.append(test_score)
			nn_batch_sizes.append(batch_size)
			nn_alphas.append(alpha)
			nn_epochs.append(epoch)
			nn_dropout.append(0)
			nn_type.append(0)
			print str(epoch) + " epochs.",\
				batch_size, "Batch Size.",\
				alpha, "Learning Rate.",\
				"Test Accuracy:\t", round(test_score, 4)
nn_results = np.column_stack((nn_type, nn_epochs, nn_batch_sizes, nn_alphas, nn_dropout, nn_accuracy))
# # save final model
# if i == len(num_epochs) - 1:
# 	pickle.dump(nn, open('nn.pkl', 'wb'))
# nn = pickle.load(open('nn.pkl', 'rb'))



num_epochs = np.arange(1, 16)
batch_sizes = [50, 100]
learning_rates = [.1, .2]
dropout_rates = [0, .25, .5]
dnn_epochs = []
dnn_accuracy = []
dnn_batch_sizes = []
dnn_alphas = []
dnn_dropouts = []
dnn_type = []
# try a deeper neural network with some better regularization
dnn_results = []
for dropout in dropout_rates:
	for alpha in learning_rates:
		for batch_size in batch_sizes:
			for epoch in num_epochs:
				dnn = Classifier(
					layers=[
						Layer("Tanh", units=200),
						Layer("Maxout", units=100, pieces=2),
						Layer("Maxout", units=50, pieces=2),
						Layer("Softmax")],
					learning_rate=alpha,
					batch_size=batch_size,
					regularize="L2",
					dropout_rate=dropout,
					n_iter=epoch,
					verbose=False)
				dnn.fit(x_train, y_train)
				test_score = dnn.score(x_test, y_test)
				dnn_epochs.append(epoch)
				dnn_accuracy.append(test_score)
				dnn_batch_sizes.append(batch_size)
				dnn_alphas.append(alpha)
				dnn_dropouts.append(dropout)
				dnn_type.append(1)
				print str(epoch) + " epochs.",\
					batch_size, "Batch Size.",\
					alpha, "Learning Rate.",\
					dropout, "Dropout Rate.",\
					"Test Accuracy:\t", round(test_score, 4)
dnn_results = np.column_stack((dnn_type, dnn_epochs, dnn_batch_sizes, dnn_alphas, dnn_dropouts, dnn_accuracy))
# # save final model
# if i == len(num_epochs) - 1:
# pickle.dump(dnn, open('dnn.pkl', 'wb'))
# dnn = pickle.load(open('dnn.pkl', 'rb'))


# save results to file
rval = np.vstack((nn_results, dnn_results))
np.savetxt("nn_results.csv", rval, delimiter=",", fmt='%10.5f')

print "null model", round(1. - y_test.mean(), 4)