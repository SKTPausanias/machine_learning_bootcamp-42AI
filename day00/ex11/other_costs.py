import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

def mse_(y, y_hat):
	"""
	Description:
	Calculate the MSE between the predicted output and the real output.
	Args:
	y: has to be a numpy.ndarray, a vector of dimension m * 1.
	y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
	Returns:
	mse: has to be a float.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exceptions.
	"""
	if len(y) < 1 or len(y_hat) < 1 or y.shape != y_hat.shape:
		return None
	return np.sum((y_hat - y) **2) / float(y.shape[0])

def rmse_(y, y_hat):
	"""
	Description:
	Calculate the RMSE between the predicted output and the real output.
	Args:
	y: has to be a numpy.ndarray, a vector of dimension m * 1.
	y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
	Returns:
	rmse: has to be a float.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exceptions.
	"""
	if len(y) < 1 or len(y_hat) < 1 or y.shape != y_hat.shape:
		return None
	return sqrt(np.sum((y_hat - y) **2) / float(y.shape[0]))

def mae_(y, y_hat):
	"""
	Description:
	Calculate the MAE between the predicted output and the real output.
	Args:
	y: has to be a numpy.ndarray, a vector of dimension m * 1.
	y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
	Returns:
	mae: has to be a float.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exceptions.
	"""
	if len(y) < 1 or len(y_hat) < 1 or y.shape != y_hat.shape:
		return None
	return np.sum(np.absolute(y_hat - y)) / float(y.shape[0])

def r2score_(y, y_hat):
	"""
	Description:
	Calculate the R2score between the predicted output and the output.
	Args:
	y: has to be a numpy.ndarray, a vector of dimension m * 1.
	y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
	Returns:
	r2score: has to be a float.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exceptions.
	"""
	if len(y) < 1 or len(y_hat) < 1 or y.shape != y_hat.shape:
		return None
	return 1 - ((np.sum((y_hat - y) **2)) / (np.sum((y_hat - np.mean(y)) **2)))
	
if __name__ == "__main__":
	x = np.array([0, 15, -9, 7, 12, 3, -21])
	y = np.array([2, 14, -13, 5, 12, 4, -19])
	print("My function:", mse_(x, y), "System function:", mean_squared_error(x, y))
	print("My function:", rmse_(x, y), "System function:", sqrt(mean_squared_error(x, y)))
	print("My function:", mae_(x, y), "System function:", mean_absolute_error(x, y))
	print("My function:", r2score_(x, y), "System function:", r2_score(x, y))