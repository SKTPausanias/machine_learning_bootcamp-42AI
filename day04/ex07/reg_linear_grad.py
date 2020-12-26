import numpy as np
from prediction import predict_
from tools import add_intercept

def reg_linear_grad(y, x, theta, lambda_):
	"""Computes the regularized linear gradient of three non-empty numpy.ndarray, with two
	,→ for-loop. The three arrays must have compatible dimensions.
	Args:
	y: has to be a numpy.ndarray, a vector of dimension m * 1.
	x: has to be a numpy.ndarray, a matrix of dimesion m * n.
	theta: has to be a numpy.ndarray, a vector of dimension n * 1.
	lambda_: has to be a float.
	Returns:
	A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all
	,→ j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	if x.size == 0 or y.size == 0 or theta.size == 0 or x.shape[0] != y.shape[0] or x is None or y is None:
		return None
	gr_vec = np.zeros((theta.shape[0], 1))
	y_hat = predict_(x, theta)
	gr_vec[0] =  np.sum((y_hat - y)) / float(y.shape[0])
	for j in range(1, theta.shape[0]):
		gr_vec[j] = (np.sum((y_hat - y) * x[:, j - 1].reshape(-1, 1)) + (lambda_ * theta[j])) / y.shape[0]
	return gr_vec

def vec_reg_linear_grad(y, x, theta, lambda_):
	"""Computes the regularized linear gradient of three non-empty numpy.ndarray, without any
	,→ for-loop. The three arrays must have compatible dimensions.
	Args:
	y: has to be a numpy.ndarray, a vector of dimension m * 1.
	x: has to be a numpy.ndarray, a matrix of dimesion m * n.
	theta: has to be a numpy.ndarray, a vector of dimension n * 1.
	lambda_: has to be a float.
	Returns:
	A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all
	,→ j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	if x.size == 0 or y.size == 0 or theta.size == 0 or x.shape[0] != y.shape[0] or x is None or y is None:
		return None
	y_hat = predict_(x, theta)
	theta2 = np.copy(theta)
	theta2[0] = 0
	gr_vec = (np.matmul(np.transpose(add_intercept(x)), (y_hat - y)) + (lambda_ * theta2)) / y.shape[0]
	return gr_vec

if __name__ == "__main__":
	x = np.array([
	[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]])
	y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
	theta = np.array([[7.01], [3], [10.5], [-6]])

	print(reg_linear_grad(y, x, theta, 1))
	print()
	print(vec_reg_linear_grad(y, x, theta, 1))
	print()

	print(reg_linear_grad(y, x, theta, 0.5))
	print()
	print(vec_reg_linear_grad(y, x, theta, 0.5))
	print()

	print(reg_linear_grad(y, x, theta, 0.0))
	print()
	print(vec_reg_linear_grad(y, x, theta, 0.0))