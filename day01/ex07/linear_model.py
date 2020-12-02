import numpy as np
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR
from sklearn.metrics import mean_squared_error
import pandas as pd

if __name__ == "__main__":
	data = pd.read_csv("../resources/are_blue_pills_magics.csv")
	Xpill = np.array(data['Micrograms']).reshape(-1,1)
	Yscore = np.array(data['Score']).reshape(-1,1)
	linear_model1 = MyLR(np.array([[89.0], [-8]]))
	linear_model2 = MyLR(np.array([[89.0], [-6]]))
	linear_model1.fit_(Xpill, Yscore)
	Y_model1 = linear_model1.predict_(Xpill)
	Y_model2 = linear_model2.predict_(Xpill)
	plt.plot(Xpill, Yscore, '.', color='b')
	plt.plot(Xpill, Y_model1, 'x', color='lime')
	plt.plot(Xpill, Y_model1, 'r--', color='lime')
	plt.xlabel('Quantity of blue pill (in micrograms)')
	plt.ylabel('Space driving score')
	plt.title	
	plt.grid()
	plt.show()
	print(linear_model1.mse_(Xpill, Yscore))
	print(mean_squared_error(Yscore, Y_model1))
	print(linear_model2.mse_(Xpill, Yscore))
	print(mean_squared_error(Yscore, Y_model2))