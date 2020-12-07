import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def cost_elem_(y_hat, y):
	"""
	Description:
	Calculates all the elements (1/M)*(y_pred - y)^2 of the cost function.
	Args:
	y: has to be an numpy.ndarray, a vector.
	y_hat: has to be an numpy.ndarray, a vector.
	Returns:
	J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
	None if there is a dimension matching problem between X, Y or theta.
	Raises:
	This function should not raise any Exception.
	"""
	return (np.power(y_hat - y, 2)) / (2 * y.shape[0])

def cost_(y_hat, y):
	"""Computes the mean squared error of two non-empty numpy.ndarray, without any for loop. The
	􏰀→ two arrays must have the same dimensions. Args:
	y: has to be an numpy.ndarray, a vector. y_hat: has to be an numpy.ndarray, a vector. Returns:
	The mean squared error of the two vectors as a float.
	None if y or y_hat are empty numpy.ndarray.
	None if y and y_hat does not share the same dimensions.
	Raises:
	This function should not raise any Exceptions.
	"""
	j_elem = cost_elem_(y_hat, y)
	return None if j_elem is None else np.sum(j_elem, dtype=float, axis=0)

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
	return 1.0 - (np.sum((y_hat - y) **2.0) / np.sum((y_hat - np.mean(y)) **2.0))

if __name__ == "__main__":
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	m_x = x.mean()
	m_y = y.mean()
	r = m_y % m_x
	f = m_y / m_x
	
	#print(r)
	#print(f)
	#lr1 = MyLinearRegression([10.71, 1.31])

	#print(lr1.predict_(x))
	#print(lr1.mse_(x, y))
	#print(lr1.cost_elem_(lr1.predict_(x), y))
	#print(lr1.cost_(lr1.predict_(x), y))

	#plt.plot(y, x, '--', color='green')
	#plt.plot(lr1.predict_(x), x, 'b', color='olive')
	#plt.show()
	#lr1.fit_(x, y)
	
	reg = LinearRegression().fit(x, y)
	#print(reg.score(x, y))
	#print(reg.coef_)
	print(reg.predict(x) / x)
	print(reg.predict(x))
	print(r2score_(reg.predict(x), y))
	print(cost_(reg.predict(x), y))
	plt.plot(y, x, '--', color='green')
	plt.plot(reg.predict(x), x, 'b', color='olive')
	plt.show()
