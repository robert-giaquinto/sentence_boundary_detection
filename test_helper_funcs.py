import numpy as np
from helper_funcs import split_validation


def test_split_validation():
	tests = [{'y': np.array([[1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1]]), 'pct_validation': 0.25}]
	expected = [[3,6,1,2]]
	results = []
	for test in tests:
		y = test['y']
		splits = split_validation(y, test['pct_validation'])
		train_ys = y[1, splits['train']]
		test_ys = y[1, splits['validation']]
		result = [sum(train_ys), len(train_ys), sum(test_ys), len(test_ys)]
		results.append(result)
	rval = 1.0 * sum([e == r for e, r in zip(expected, results)]) / len(expected)
	return rval


x = test_split_validation()
print "split_validation:", x
