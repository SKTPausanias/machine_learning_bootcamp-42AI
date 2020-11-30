from prediction import predict_
from vec_cost import cost_
import numpy as np
import matplotlib.pyplot as plt

def plot_with_cost(x, y, theta):
	"""Plot the data and prediction line from three non-empty numpy.ndarray.
	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
	Nothing.
	Raises:
	This function should not raise any Exception.
	"""
	y_hat = predict_(x, theta)
	plt.title(f'Cost : {cost_(y, y_hat)}')
	for i in range(x.shape[0]):
		plt.plot([x[i], x[i]], [y[i], y_hat[i]], 'r--', color='r')
	plt.plot(x, y, '.')
	plt.plot(x, y_hat)

	plt.show()

if __name__ == "__main__":
	x = np.arange(1,6)
	y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
	#Example 1:
	theta1= np.array([18,-1])
	plot_with_cost(x, y, theta1)

	theta2 = np.array([14, 0])
	plot_with_cost(x, y, theta2)

	theta3 = np.array([12, 0.8])
	plot_with_cost(x, y, theta3)