import numpy as np
from prediction import predict_
from tools import add_intercept

def vec_gradient(x, y, theta):
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
	y_hat = predict_(x, theta)
	gr_vec = (np.matmul(add_intercept(x).transpose(), (y_hat - y))) / y.shape[0]
	return gr_vec

if __name__ == "__main__":
	X = np.array([
		[ -6,  -7,  -9],
			[ 13,  -2,  14],
			[ -7,  14,  -1],
			[ -8,  -4,   6],
			[ -5,  -9,   6],
			[  1,  -5,  11],
			[  9, -11,   8]])
	Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
	theta = np.array([[0], [3], [0.5], [-6]])
	print(vec_gradient(X, Y, theta))
	# array([ -37.35714286, 183.14285714, -393.])

	theta = np.array([[0], [0], [0], [0]])
	print(vec_gradient(X, Y, theta))
	# array([  0.85714286, 23.28571429, -26.42857143])

	print(vec_gradient(X, add_intercept(X).dot(theta), theta))
	# array([0., 0., 0.])