import numpy as np
from gradient import gradient
from prediction import predict_

def fit_(x, y, theta, alpha, n_cycles):
	"""
	Description:
	Fits the model to the training dataset contained in x and y.
	Args:
	x: has to be a numpy.ndarray, a matrix of dimension m * n: (number of training examples, 􏰀→ number of features).
	y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
	theta: has to be a numpy.ndarray, a vector of dimension (n + 1) * 1: (number of features +
	􏰀→ 1, 1).
	alpha: has to be a float, the learning rate
	n_cycles: has to be an int, the number of iterations done during the gradient descent Returns:
	new_theta: numpy.ndarray, a vector of dimension (number of features + 1, 1).
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exception.
	"""
	if len(x) < 1 or len(y) < 1 or len(theta) < 1 or x.shape[0] != y.shape[0] or x is None or y is None:
		return None
	for _ in range(n_cycles):
		theta -= (gradient(x, y, theta) * alpha)
	return theta

if __name__ == "__main__":
	x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
	y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
	theta = np.array([[42.], [1.], [1.], [1.]])
	# Example 0:
	theta2 = fit_(x, y, theta,  alpha = 0.0005, n_cycles=42000)
	print(theta2)
	print(predict_(x, theta2))