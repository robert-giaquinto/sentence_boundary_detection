from __future__ import division
import pickle
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sknn.mlp import Classifier, Layer
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold

data_dir = '/Users/robert/Documents/UMN/5551_Robots/Project/data/'
df = pd.read_csv(data_dir + 'training.csv')
x = df[df.columns[3:]]
y = df[df.columns[2]]

# split intro test and training
x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=2015)

# Normalize data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_test = scaler.transform(x_test)

# pca transformation on review variables
# pca = PCA(n_components=30, whiten=True)
# num_real_vars = 87
# x_train_pca = pca.fit_transform(x_train[:, num_real_vars:])
# x_train = np.hstack((x_train[:, 0:num_real_vars], x_train_pca))
# x_test_pca = pca.transform(x_test[:, num_real_vars:])
# x_test = np.hstack((x_test[:, 0:num_real_vars], x_test_pca))



# create neural network with same architecture are my code
num_epochs = np.arange(1, 16)
batch_sizes = [100, 1000]
learning_rates = [.01, .1]
dropout_rates = [0, 0.5]

nn_accuracy = []
nn_batch_sizes = []
nn_alphas = []
nn_num_units = []
nn_dropout = []
nn_epochs = []
nn_type = []
nn_fold = []
skf = StratifiedKFold(y, n_folds=3)
fold = 0
for train_index, valid_index in skf:
	fold += 1
	x_train, x_valid = x.values[train_index], x.values[valid_index]
	y_train, y_valid = y.values[train_index], y.values[valid_index]
	# Normalize data
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_valid = scaler.transform(x_valid)
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
				valid_score = nn.score(x_valid, y_valid)
				nn_accuracy.append(valid_score)
				nn_batch_sizes.append(batch_size)
				nn_alphas.append(alpha)
				nn_epochs.append(epoch)
				nn_dropout.append(0)
				nn_type.append(0)
				nn_fold.append(fold)
				print str(epoch) + " epochs.",\
					batch_size, "Batch Size.",\
					alpha, "Learning Rate.",\
					"Fold", fold,\
					"Valid Accuracy:\t", round(valid_score, 4)

nn_results = np.column_stack((nn_type, nn_epochs, nn_batch_sizes, nn_alphas, nn_dropout, nn_accuracy, nn_fold))
nn_df = pd.DataFrame(nn_results[:, 1:], columns=['epoch', 'batch_size', 'alpha', 'dropout', 'accuracy', 'fold'])
nn_final = nn_df.loc[nn_df.epoch == np.max(num_epochs), ['epoch', 'batch_size', 'alpha', 'dropout', 'accuracy']]
nn_agg = nn_final.groupby(['epoch', 'batch_size', 'alpha', 'dropout']).mean().reset_index()
best = nn_agg.loc[nn_agg.accuracy == np.max(nn_agg.accuracy), :]
print "BEST MLP CV:", best

nn = Classifier(
	layers=[
		Layer("Tanh", units=100),
		Layer("Softmax")],
	learning_rate=float(best.alpha[0]),
	batch_size=int(best.batch_size[0]),
	regularize="L2",
	n_iter=np.max(num_epochs),
	verbose=False)
nn.fit(x_scaled, y)
test_score = nn.score(x_test, y_test)
nn_accuracy.append(test_score)
nn_batch_sizes.append(best.batch_size[0])
nn_alphas.append(best.alpha[0])
nn_epochs.append(np.max(num_epochs))
nn_dropout.append(0)
nn_type.append(0)
nn_fold.append(0)
nn_results = np.column_stack((nn_type, nn_epochs, nn_batch_sizes, nn_alphas, nn_dropout, nn_accuracy, nn_fold))
# save final model
pickle.dump(nn, open('nn.pkl', 'wb'))
# nn = pickle.load(open('nn.pkl', 'rb'))






# try a deeper neural network with some better regularization
print "\nSTARTING DEEP LEARNING\n"

dnn_epochs = []
dnn_accuracy = []
dnn_batch_sizes = []
dnn_alphas = []
dnn_dropout = []
dnn_type = []
dnn_fold = []
dnn_results = []
fold = 0
for train_index, valid_index in skf:
	fold += 1
	x_train, x_valid = x.values[train_index], x.values[valid_index]
	y_train, y_valid = y.values[train_index], y.values[valid_index]
	# Normalize data
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_valid = scaler.transform(x_valid)

	for dropout in dropout_rates:
		for alpha in learning_rates:
			for batch_size in batch_sizes:
				for epoch in num_epochs:
					dnn = Classifier(
						layers=[
							Layer("Tanh", units=300),
							Layer("Maxout", units=200, pieces=2),
							Layer("Tanh", units=100),
							Layer("Maxout", units=50, pieces=2),
							Layer("Softmax")],
						learning_rate=alpha,
						batch_size=batch_size,
						regularize="L2",
						dropout_rate=dropout,
						n_iter=epoch,
						verbose=False)
					dnn.fit(x_train, y_train)
					valid_score = dnn.score(x_valid, y_valid)
					dnn_epochs.append(epoch)
					dnn_accuracy.append(test_score)
					dnn_batch_sizes.append(batch_size)
					dnn_alphas.append(alpha)
					dnn_dropout.append(dropout)
					dnn_type.append(1)
					dnn_fold.append(fold)
					print str(epoch) + " epochs.",\
						batch_size, "Batch Size.",\
						alpha, "Learning Rate.",\
						dropout, "Dropout Rate.",\
						"Fold", fold,\
						"Valid Accuracy:\t", round(valid_score, 4)
dnn_results = np.column_stack((dnn_type, dnn_epochs, dnn_batch_sizes, dnn_alphas, dnn_dropout, dnn_accuracy, dnn_fold))
dnn_df = pd.DataFrame(dnn_results[:, 1:], columns=['epoch', 'batch_size', 'alpha', 'dropout', 'accuracy', 'fold'])
dnn_final = dnn_df.loc[dnn_df.epoch == np.max(num_epochs), ['epoch', 'batch_size', 'alpha', 'dropout', 'accuracy']]
dnn_agg = dnn_final.groupby(['epoch', 'batch_size', 'alpha', 'dropout']).mean().reset_index()
best = dnn_agg.loc[dnn_agg.accuracy == np.max(dnn_agg.accuracy), :]
print "BEST MLP CV:", best

dnn = Classifier(
	layers=[Layer("Tanh", units=300),
		Layer("Maxout", units=200, pieces=2),
		Layer("Tanh", units=100),
		Layer("Maxout", units=50, pieces=2),
		Layer("Softmax")],
	learning_rate=float(best.alpha[0]),
	batch_size=int(best.batch_size[0]),
	regularize="L2",
	dropout_rate=float(best.dropout[0]),
	n_iter=np.max(num_epochs),
	verbose=False)
dnn.fit(x_scaled, y)
test_score = dnn.score(x_test, y_test)
dnn_accuracy.append(test_score)
dnn_epochs.append(np.max(num_epochs))
dnn_batch_sizes.append(int(best.batch_size[0]))
dnn_alphas.append(float(best.alpha[0]))
dnn_dropout.append(float(best.dropout[0]))
dnn_type.append(1)
dnn_fold.append(0)
dnn_results = np.column_stack((dnn_type, dnn_epochs, dnn_batch_sizes, dnn_alphas, dnn_dropout, dnn_accuracy, dnn_fold))
# save final model
pickle.dump(dnn, open('dnn.pkl', 'wb'))
# nn = pickle.load(open('nn.pkl', 'rb'))

# save results to file
rval = np.vstack((nn_results, dnn_results))
np.savetxt("results_final.csv", rval, delimiter=",", fmt='%10.5f')

print "null model", round(y_test.mean(), 4)