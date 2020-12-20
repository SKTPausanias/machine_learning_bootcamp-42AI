import numpy as np
import matplotlib.pyplot as plt

class MyLogisticRegression():
	"""
	Description:
	My personnal linear regression class to fit like a boss.
	"""
	def __init__(self,  thetas, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas

	def gradient(self, x, y, theta):
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
		if len(x) < 1 or len(y) < 1 or len(self.thetas) < 1 or x is None or y is None or self.thetas is None or x.shape[0] != y.shape[0]:
			return None
		y_hat = self.predict_(x)
		gr_vec = (np.matmul(np.transpose(self.add_intercept(x)), (y_hat - y))) / y.shape[0]
		return gr_vec

	def fit_(self, x, y):
		"""
		Description:
		Fits the model to the training dataset contained in x and y.
		Args:
		x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
		y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
		theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
		alpha: has to be a float, the learning rate
		max_iter: has to be an int, the number of iterations done during the gradient descent
		Returns:
		new_theta: numpy.ndarray, a vector of dimension 2 * 1.
		None if there is a matching dimension problem.
		Raises:
		This function should not raise any Exception.
		"""
		if len(x) < 1 or len(y) < 1 or len(self.thetas) < 1 or x.shape[0] != y.shape[0] or x is None or y is None:
			return None
		#x_norm = np.zeros(x.shape)
		#i = 0
		#while i < x.shape[1]:
		#	x_norm[:,i] = (x[:,i] - x[:,i].mean()) / x[:,i].std()
		#	i += 1
		#y_norm = (y - y.mean()) / y.std()
		for _ in range(self.max_iter):
			self.thetas -= (self.gradient(x, y, self.thetas) * self.alpha)
		#res = self.thetas[0]
		#i = 1
		#while i < len(self.thetas):
		#	res -= (self.thetas[i] * x[:,i - 1].mean() / x[:,i - 1].std())
		#	i += 1
		#self.thetas[0] = (res * y.std()) + y.mean()
		#i = 1
		#while i < len(self.thetas):
		#	self.thetas[i] = (self.thetas[i] * y.std() / x[:,i - 1].std())
		#	i += 1
		return self.thetas
	
	def predict_(self, x):
		"""Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
		Args:
		x: has to be an numpy.ndarray, a vector of dimension m * n.
		theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
		Returns:
		y_hat as a numpy.ndarray, a vector of dimension m * 1.
		None if x or theta are empty numpy.ndarray.
		None if x or theta dimensions are not appropriate.
		Raises:
		This function should not raise any Exception.
		"""
		if len(x) < 1 or len(self.thetas) < 1 or x is None or self.thetas is None:
			return None
		return self.sigmoid_(np.matmul(self.add_intercept(x), self.thetas))

	def sigmoid_(self, x):
		"""
		Compute the sigmoid of a vector.
		Args:
		x: has to be an numpy.ndarray, a vector
		Returns:
		The sigmoid value as a numpy.ndarray.
		None if x is an empty numpy.ndarray.
		Raises:
		This function should not raise any Exception.
		"""
		if x.size == 0 or x is None:
			return None
		return 1 / (1 + np.exp(-x))

	def add_intercept(self, x):
		"""Adds a column of 1's to the non-empty numpy.ndarray x.
		Args:
		x: has to be an numpy.ndarray, a vector of dimension m * 1.
		Returns:
		X as a numpy.ndarray, a vector of dimension m * 2.
		None if x is not a numpy.ndarray.
		None if x is a empty numpy.ndarray.
		Raises:
		This function should not raise any Exception.
		"""
		if len(x) < 1 or type(x) is not np.ndarray:
			return None
		return np.c_[np.ones(x.shape[0]), x]

	def cost_(self, y_hat, y, eps=1e-16):
		"""
		Computes the logistic loss value.
		Args:
		y: has to be an numpy.ndarray, a vector of dimension m * 1.
		y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
		eps: has to be a float, epsilon (default=1e-15)
		Returns:
		The logistic loss value as a float.
		None on any error.
		Raises:
		This function should not raise any Exception.
		"""
		return -(np.sum(((y + eps) * np.log(y_hat)) + ((1 - (y + eps)) * np.log(1 - y_hat)))) * (1 / y.shape[0])

if __name__ == "__main__":
	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
	Y = np.array([[1], [0], [1]])
	mylr = MyLogisticRegression([[2], [0.5], [7.1], [-4.3], [2.09]])

	print(mylr.predict_(X))
	print(mylr.cost_(mylr.predict_(X), Y))

	#mylr.fit_(X, Y)
	#print(mylr.thetas)

	#print(mylr.predict_(X))
	#print(mylr.cost_(X,Y))
