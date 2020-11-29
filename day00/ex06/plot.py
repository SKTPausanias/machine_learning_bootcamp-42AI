from prediction import predict_
import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, theta):
	"""Plot the data and prediction line from three non-empty numpy.ndarray.
	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
	Nothing.
	Raises:
	This function should not raise any Exceptions.
	"""
	plt.plot(x, y, '.')
	plt.plot(x, predict_(x, theta))
	plt.show()

if __name__ == "__main__":
	x = np.arange(1,6)
	y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
	theta1 = np.array([4.5, -0.2])
	plot(x, y, theta1)
	theta1 = np.array([-1.5, 2])
	plot(x, y, theta1)
	theta1 = np.array([3, 0.3])
	plot(x, y, theta1)