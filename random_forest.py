from __future__ import division
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pyparsing
import matplotlib.pyplot as plt
import time

# implement a standard random forest algorithm to evaluate current feature set

data_dir = '/Users/robert/Documents/UMN/5551_Robots/Project/data/'
df = pd.read_csv(data_dir + 'training.csv')
x = df[df.columns[3:]]
y = df[df.columns[2]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=2015)

forest = RandomForestClassifier(n_estimators=100, max_features="sqrt", n_jobs=1)
forest = forest.fit(x_train, y_train)
forest.score(x_train, y_train)
forest.score(x_test, y_test)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(x.shape[1]):
	print("%d. feature %s (%f)" % (f + 1, x.columns[indices[f]], importances[indices[f]]))



# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), indices)
plt.xlim([-1, x.shape[1]])
plt.show()