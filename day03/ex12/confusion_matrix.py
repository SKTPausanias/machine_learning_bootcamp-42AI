import numpy as np
from sklearn.metrics import confusion_matrix

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
	#lookupTable2, indexed_dataSet2 = np.unique(y_hat, return_inverse=True)
	#lookupTable, indexed_dataSet = np.unique(y_true, return_inverse=True)
	#print(y_true)
	#print(indexed_dataSet)
	#print(y_hat)
	#print(indexed_dataSet2)
	#K = len(np.unique(y_hat)) # Number of classes 
	#result = np.zeros((K, K))
#
	#for i in range(len(y_true)):
	#	result[indexed_dataSet[i]][indexed_dataSet2[i]] += 1
#
	#return result

if __name__ == "__main__":
	y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
	y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])

	print(confusion_matrix_(y, y_hat))