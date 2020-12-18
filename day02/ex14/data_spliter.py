import numpy as np
import pandas as pd
from my_linear_regression import MyLinearRegression
from polynomial_model import add_polynomial_features

def data_spliter(x, y, proportion):
	"""Shuffles and splits the dataset (given by x and y) into a training and a test set, while
	,→ respecting the given proportion of examples to be kept in the traning set.
	Args:
	x: has to be an numpy.ndarray, a matrix of dimension m * n.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	proportion: has to be a float, the proportion of the dataset that will be assigned to the
	,→ training set.
	Returns:
	(x_train, x_test, y_train, y_test) as a tuple of numpy.ndarray
	None if x or y is an empty numpy.ndarray.
	None if x and y do not share compatible dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	if type(proportion) is not float:
		return None
	if x.shape == (x.shape[0],):
		x = x[:, np.newaxis]
	if y.shape == (y.shape[0],):
		y = y[:, np.newaxis]
	n = np.concatenate((x, y), axis=1)
	np.random.shuffle(n)
	sep = int(n.shape[0]* proportion)
	array1 = n[:sep , :] # indexing/selection of the 80%
	array2 = n[sep: , :]
	return(array1[:,:-1], array2[:,:-1], array1[:, -1], array2[:, -1])

if __name__ == "__main__":
	data = pd.read_csv("../resources/are_blue_pills_magics.csv")
	x = np.array(data[['Micrograms']])
	y = np.array(data[['Score']])

	lst = data_spliter(x, y, 0.5)
	x_train = lst[0]
	y_train = lst[2]
	y_train = y_train[:, np.newaxis]
	x_test = lst[1]
	y_test = lst[3]
	y_test = y_test[:, np.newaxis]

	i = 2
	my_lr = MyLinearRegression([[1],[1]])
	my_lr.fit_(x_train, y_train)
	y_hat = my_lr.predict_(x_test)
	print(my_lr.cost_(y_hat, y_test))

	while i <= 10:
		x_ = add_polynomial_features(x_train, i)
		my_lr = MyLinearRegression(np.ones(i + 1).reshape(-1,1))
		my_lr.fit_(x_, y_train)
		x_2 = add_polynomial_features(x_test, i)
		y_hat = my_lr.predict_(x_2)
		print(my_lr.cost_(y_hat, y_test))
		i += 1