import numpy as np
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR
from sklearn.metrics import mean_squared_error
import pandas as pd

def predict_(x: np.ndarray, theta: np.ndarray):
	if len(x) < 1 or theta.shape != (2,):
		return None
	return theta[0] + (theta[1] * x)

def add_intercept(x):
	if len(x) < 1 or type(x) is not np.ndarray:
		return None
	return np.c_[np.ones(x.shape[0]), x]


def cost_elem_(y_hat, y):
	return (np.power(y_hat - y, 2)) / (2 * y.shape[0])

def cost_(y_hat, y):
	j_elem = cost_elem_(y_hat, y)
	return np.sum(j_elem)

if __name__ == "__main__":
	data = pd.read_csv("../resources/are_blue_pills_magics.csv")
	Xpill = np.array(data['Micrograms']).reshape(-1,1)
	Yscore = np.array(data['Score']).reshape(-1,1)
	linear_model1 = MyLR(np.array([[89.0], [-8]]))
	linear_model2 = MyLR(np.array([[89.0], [-6]]))

	Y_model1 = linear_model1.predict_(Xpill)
	Y_model2 = linear_model2.predict_(Xpill)
	print(linear_model1.mse_(Xpill, Yscore))
	print(mean_squared_error(Yscore, Y_model1))
	print(linear_model2.mse_(Xpill, Yscore))
	print(mean_squared_error(Yscore, Y_model2))

	print(Yscore)
	#print(linear_model1.cost_(linear_model1.predict_(Xpill), Yscore))
	#linear_model1.fit_(Xpill, Yscore)
	print(linear_model1.thetas)
	theta1_array = np.array([-14, -12, -10, -8, -6, -4])
	cost_array = np.zeros(6,)
	for j in range(87, 92):
		for i in range(6):
			print(theta1_array[i])
			pre = predict_(Xpill, np.array([j, theta1_array[i]]))
			#print(pre)
			#print(cost_(pre, Yscore))
			cost_array[i] = cost_(pre, Yscore)
		plt.plot(theta1_array, cost_array)
	print(cost_array)
	#plt.plot(theta1_array, cost_array)
	plt.grid()
	plt.show()

	linear_model1.fit_(Xpill, Yscore)
	Y_model1 = linear_model1.predict_(Xpill)	
	plt.plot(Xpill, Yscore, '.', color='b')
	plt.plot(Xpill, Y_model1, 'x', color='lime')
	plt.plot(Xpill, Y_model1, 'r--', color='lime')
	plt.xlabel('Quantity of blue pill (in micrograms)')
	plt.ylabel('Space driving score')
	plt.title	
	plt.grid()
	plt.show()