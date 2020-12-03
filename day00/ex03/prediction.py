import numpy as np

def simple_predict(x: np.ndarray, theta: np.ndarray):
	"""Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
	y_hat as a numpy.ndarray, a vector of dimension m * 1.
	None if x or theta are empty numpy.ndarray.
	None if x or theta dimensions are not appropriate.
	Raises:
	This function should not raise any Exception.
	"""
	if len(x) < 1 or theta.shape != (2,):
		return None
	return theta[0] + (theta[1] * x)

if __name__ == "__main__":
	x = np.arange(1,6)
	theta1 = np.array([5, 0])
	print(simple_predict(x, theta1))
	theta2 = np.array([0, 1])
	print(simple_predict(x, theta2))
	theta3 = np.array([5, 3])
	print(simple_predict(x, theta3))
	theta4 = np.array([-3, 1])
	print(simple_predict(x, theta4))