import numpy as np
from prediction import simple_predict
from tools import add_intercept

def gradient(x, y, theta):
	"""Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The
	􏰀→ three arrays must have the compatible dimensions.
	Args:
	x: has to be an numpy.ndarray, a matrix of dimension m * n.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a vector (n +1) * 1.
	Returns:
	The gradient as a numpy.ndarray, a vector of dimensions n * 1, containg the result of the
	􏰀→ formula for all j.
	None if x, y, or theta are empty numpy.ndarray.
	None if x, y and theta do not have compatible dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	if len(x) < 1 or len(y) < 1 or len(theta) < 1 or x is None or y is None or theta is None or x.shape[0] != y.shape[0]:
		return None
	y_hat = simple_predict(x, theta)
	gr_vec = (np.matmul(add_intercept(x).transpose(), (y_hat - y))) / y.shape[0]
	return gr_vec
