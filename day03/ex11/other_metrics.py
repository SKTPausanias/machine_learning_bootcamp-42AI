import numpy as np

def accuracy_score_(y, y_hat):
	"""
	Compute the accuracy score.
	Args:
	y:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels
	Returns:
	The accuracy score as a float.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
	score = y == y_hat
	return np.average(score)

def precision_score_(y, y_hat, pos_label=1):
	"""
	Compute the precision score.
	Args:
	y:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels
	pos_label: str or int, the class on which to report the precision_score (default=1)
	Returns:
	The precision score as a float.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
	tp = 0
	fp = 0
	for i in range(len(y)):
		if y[i] == pos_label and y_hat[i] == y[i]:
			tp += 1
		elif y[i] != pos_label and y_hat[i] == pos_label:
			fp += 1
	return (tp / (tp + fp))


def recall_score_(y, y_hat, pos_label=1):
	"""
	Compute the recall score.
	Args:
	y:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels
	pos_label: str or int, the class on which to report the precision_score (default=1)
	Returns:
	The recall score as a float.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
	tp = 0
	fn = 0
	for i in range(len(y)):
		if y[i] == pos_label and y_hat[i] == y[i]:
			tp += 1
		elif y[i] == pos_label and y_hat[i] != pos_label:
			fn += 1
	return (tp / (tp + fn))

def f1_score_(y, y_hat, pos_label=1):
	"""
	Compute the f1 score.
	Args:
	y:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels
	pos_label: str or int, the class on which to report the precision_score (default=1)
	Returns:
	The f1 score as a float.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
	prec = precision_score_(y, y_hat, pos_label)
	rec = recall_score_(y, y_hat, pos_label)
	return (2 * prec * rec) / (prec + rec)

if __name__ == "__main__":
	y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1])
	y = np.array([1, 0, 0, 1, 0, 1, 0, 0])

	print(accuracy_score_(y, y_hat))
	print(precision_score_(y, y_hat))
	print(recall_score_(y, y_hat))
	print(f1_score_(y, y_hat))

	y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
	y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

	print(accuracy_score_(y, y_hat))
	print(precision_score_(y, y_hat, pos_label='dog'))
	print(recall_score_(y, y_hat, pos_label='dog'))
	print(f1_score_(y, y_hat, pos_label='dog'))

	y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
	y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

	print(accuracy_score_(y, y_hat))
	print(precision_score_(y, y_hat, pos_label='norminet'))
	print(recall_score_(y, y_hat, pos_label='norminet'))
	print(f1_score_(y, y_hat, pos_label='norminet'))