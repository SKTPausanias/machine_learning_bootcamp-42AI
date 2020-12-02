import numpy as np
from prediction import predict_
from tools import add_intercept

def gradient(x, y, theta):
	"""Computes a gradient vector from three non-empty numpy.ndarray, without any for loop. The
	,→ three arrays must have compatible dimensions.
	Args:
	x: has to be a numpy.ndarray, a matrix of dimension m * 1.
	y: has to be a numpy.ndarray, a vector of dimension m * 1.
	theta: has to be a numpy.ndarray, a 2 * 1 vector.
	Returns:
	The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
	None if x, y, or theta is an empty numpy.ndarray.
	None if x, y and theta do not have compatible dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	if len(x) < 1 or len(y) < 1 or len(theta) < 1 or x.shape != y.shape or theta.shape != (2,) or x is None or y is None:
		return None
	y_hat = predict_(x, theta)
	gr_vec = (np.matmul(np.transpose(add_intercept(x)), (y_hat - y))) / y.shape[0]
	return gr_vec


if __name__ == "__main__":
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])

	theta1 = np.array([2, 0.7])
	print(gradient(x, y, theta1))

	theta2 = np.array([1, -0.4])
	print(gradient(x, y, theta2))