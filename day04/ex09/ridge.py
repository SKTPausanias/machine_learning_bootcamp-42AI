import numpy as np

class MyRidge():
	"""
	Description:
	My personnal ridge regression class to fit like a boss.
	"""
	def __init__(self, thetas, alpha=0.001, n_cycle=1000, lambda_=0.5):
		self.alpha = alpha
		self.n_cycle = n_cycle
		self.thetas = thetas
		self.lambda_ = lambda_

	def gradient(self, x, y):
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
		if x.size == 0 or y.size == 0 or self.thetas.size == 0 or x.shape[0] != y.shape[0] or x is None or y is None:
			return None
		y_hat = self.predict_(x)
		thetas2 = np.copy(self.thetas)
		thetas2[0] = 0
		gr_vec = (np.matmul(np.transpose(self.add_intercept(x)), (y_hat - y)) + (self.lambda_ * thetas2)) / y.shape[0]
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
		x_norm = np.zeros(x.shape)
		i = 0
		while i < x.shape[1]:
			x_norm[:,i] = (x[:,i] - x[:,i].mean()) / x[:,i].std()
			i += 1
		y_norm = (y - y.mean()) / y.std()
		for _ in range(self.n_cycle):
			self.thetas -= (self.gradient(x_norm, y_norm) * self.alpha)
		res = self.thetas[0]
		i = 1
		while i < len(self.thetas):
			res -= (self.thetas[i] * x[:,i - 1].mean() / x[:,i - 1].std())
			i += 1
		self.thetas[0] = (res * y.std()) + y.mean()
		i = 1
		while i < len(self.thetas):
			self.thetas[i] = (self.thetas[i] * y.std() / x[:,i - 1].std())
			i += 1
		return self.thetas
	
	def predict_(self, x):
		"""Computes the prediction vector y_hat from two non-empty numpy.ndarray.
		Args:
		x: has to be an numpy.ndarray, a matrix of dimension m * n.
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
		return np.matmul(self.add_intercept(x), self.thetas)

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

	def reg_cost_(self, y, y_hat):
		"""Computes the regularized cost of a linear regression model from two non-empty
		,→ numpy.ndarray, without any for loop. The two arrays must have the same dimensions.
		Args:
		y: has to be an numpy.ndarray, a vector of dimension m * 1.
		y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
		theta: has to be a numpy.ndarray, a vector of dimension n * 1.
		lambda_: has to be a float.
		Returns:
		The regularized cost as a float.
		None if y, y_hat, or theta are empty numpy.ndarray.
		None if y and y_hat do not share the same dimensions.
		Raises:
		This function should not raise any Exception.
		"""
		if y.size == 0 or y_hat.size == 0 or self.thetas.size == 0 or y.shape != y_hat.shape:
			return None
		thetas2 = np.copy(self.thetas)
		thetas2[0] = 0
		return (np.dot((y_hat - y), (y_hat - y)) + (self.lambda_ * (np.dot(thetas2, thetas2)))) / float(y.shape[0] * 2) 
