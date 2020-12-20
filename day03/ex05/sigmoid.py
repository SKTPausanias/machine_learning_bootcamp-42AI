import numpy as np
import math

def sigmoid_(x):
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

if __name__ == "__main__":

	x = np.array(-4)
	print(sigmoid_(x))

	x = np.array(2)
	print(sigmoid_(x))

	x = np.array([[-4], [2], [0]])
	print(sigmoid_(x))