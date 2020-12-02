import numpy as np
from tools import add_intercept

def predict_(x, theta) -> np.ndarray: 
	"""Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
	y_hat as a numpy.ndarray, a vector of dimension m * 1.
	None if x or theta are empty numpy.ndarray.
	None if x or theta dimensions are not appropriate.
	Raises:
	This function should not raise any Exceptions.
	"""
	#theta = np.reshape(theta, (theta.shape[0],))
	if len(x) < 1 and theta.shape == (theta.shape[0],):
		return None
	#return np.sum(add_intercept(x) * theta, axis=1)
	return np.matmul(add_intercept(x), theta)

if __name__ == "__main__":
	x = np.arange(1,6)
	theta1 = np.array([5, 0])
	print(theta1.shape)
	print(predict_(x, theta1))
	theta2 = np.array([0, 1])
	print(predict_(x, theta2))
	theta3 = np.array([5, 3])
	print(predict_(x, theta3))
	theta4 = np.array([-3, 1])
	print(predict_(x, theta4))