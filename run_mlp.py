from __future__ import division
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from MLP import MLP
import logging

# import data
data_dir = '/Users/robert/Documents/UMN/5551_Robots/Project/data/'
df = pd.read_csv(data_dir + 'training.csv')
x = df[df.columns[3:28]].values
y = df[df.columns[2]].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=2015)

# Normalize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# prepare data: transpose x's and y's, and binary-ize the y's
x_train = x_train.T
x_test = x_test.T
y_train = np.vstack((1 - y_train, y_train))
y_test = np.vstack((1 - y_test, y_test))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
mlp = MLP(num_hidden_units=50, alpha=.01, lambda_penalty=0.0001, activation="tanh", random_seed=1234)
mlp.fit(x_train, y_train, epochs=50, batch_size=10000, stall_limit=100, pct_validation=0.1)

probs_train = mlp.predict_proba(x_train)
loss = np.mean((probs_train - y_train)**2)
print "Training Loss: ", loss

probs_test = mlp.predict_proba(x_test)
loss = np.mean((probs_test - y_test)**2)
print "Test Loss: ", loss

class_scores = mlp.score_classes(x_train, y_train)
print "Train class scores:", class_scores
class_scores = mlp.score_classes(x_test, y_test)
print "Test class scores:", class_scores

print "Train Score"
print mlp.score(x_train, y_train)

print "Test Score"
print mlp.score(x_test, y_test)