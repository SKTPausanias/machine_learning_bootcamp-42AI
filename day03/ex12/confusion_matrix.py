import numpy as np

def confusion_matrix_(y_true, y_hat, labels=None):
	"""
	Compute confusion matrix to evaluate the accuracy of a classification.
	Args:
	y_true:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels
	labels: optional, a list of labels to index the matrix. This may be used to reorder or
	,→ select a subset of labels. (default=None)
	Returns:
	The confusion matrix as a numpy ndarray.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
	if labels is not None:
		for i in range(len(y_hat)):
			if y_hat[i] not in labels or y_true[i] not  in labels:
				y_hat = np.delete(y_hat, i)
				y_true = np.delete(y_true, i)
	conc = np.concatenate((y_hat, y_true))
	_, indexed_dataSet = np.unique(conc, return_inverse=True)
	K = len(np.unique(indexed_dataSet))
	splited = np.split(indexed_dataSet, 2)
	y_hat = splited[0]
	y_true = splited[1]
	
	result = np.zeros((K, K))

	for i in range(len(y_hat)):
		result[y_true[i]][y_hat[i]] += 1
	return result

if __name__ == "__main__":
	y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
	y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])

	#confusion_matrix_(y, y_hat)
	print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
	print(confusion_matrix_(y, y_hat))