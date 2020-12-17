import numpy as np
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression
from polynomial_model import add_polynomial_features

if __name__ == "__main__":
	x = np.arange(1,11).reshape(-1,1)
	y = np.array([[ 1.39270298],
	[ 3.88237651],
	[ 4.37726357],
	[ 4.63389049],
	[ 7.79814439],
	[ 6.41717461],
	[ 8.63429886],
	[ 8.19939795],
	[10.37567392],
	[10.68238222]])
	plt.scatter(x,y)
	plt.show()

	i = 2
	arr = np.zeros(9)
	l = list(range(2, 11))
	while i <= 10:
		x_ = add_polynomial_features(x, i)
		my_lr = MyLinearRegression(np.ones(i + 1).reshape(-1,1))
		my_lr.fit_(x_, y)
		arr[i - 2 ] = (my_lr.cost_(my_lr.predict_(x_), y))

		continuous_x = np.arange(1,10.01, 0.01).reshape(-1,1)
		x_ = add_polynomial_features(continuous_x, i)
		y_hat = my_lr.predict_(x_)

		plt.scatter(x,y)
		plt.plot(continuous_x, y_hat, color='orange')
		plt.show()
		i += 1
	plt.bar(l, arr, color='orange')
	plt.show()
	print(arr)