#!/usr/bin/python3

from matrix import Matrix
import numpy as np

if __name__ == "__main__":
	lst = [[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]]

	m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
					[0.0, 2.0, 4.0, 6.0]])

	m2 = Matrix([[3.0, -1.0, 4.0, -3.0],
					[2.0, 1.0, 4.0, 2.0]])

	m3 = Matrix((3, 3))

	m4 = Matrix([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], (3, 3))

	print(m1)
	print(m3)
	print(m4)

	print(m1 + m2)
	print(m1 - m2)
	print(m1 / 2)

	X = Matrix([[12.0, 7.0, 3.0], [4.0, 5.0, 6.0], [7.0 ,8.0,9.0]])
	# 3x4 matrix
	Y = Matrix([[5.0,8.0,1.0,2.0], [6.0,7.0,3.0,0.0], [4.0,5.0,9.0,1.0]])

	x = Matrix([[1.0, -2.0, 3.0]])
	y = Matrix([[4.0], [5.0], [6.0]])

	print(X * Y)
	print(x * y)