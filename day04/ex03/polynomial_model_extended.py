import numpy as np

def add_polynomial_features(x, power):
	"""Add polynomial features to matrix x by raising its columns to every power in the range of
	,→ 1 up to the power given in argument.
	Args:
	x: has to be an numpy.ndarray, a matrix of dimension m * n.
	power: has to be an int, the power up to which the columns of matrix x are going to be
	,→ raised.
	Returns:
	The matrix of polynomial features as a numpy.ndarray, of dimension m * (np), containg the
	,→ polynomial feature values for all training examples.
	None if x is an empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""
	ret = np.zeros((x.shape[0], x.shape[1] * power))
	for i in range(x.shape[0]):
		j = 1
		h = 1
		while j <= power:
			ret[i][h - 1:h - 1 + x.shape[1]] = np.power(x[i], j)
			j += 1
			h += x.shape[1]
	return ret

if __name__ == "__main__":
	x = np.arange(1,11).reshape(5, 2)
	np.set_printoptions(suppress=True)
	print(add_polynomial_features(x, 3))
	print(add_polynomial_features(x, 4))
	print(add_polynomial_features(x, 5))
