#!/usr/bin/python3

from vector import Vector

if __name__ == "__main__":
	v = Vector([0.2, 1.3, 4])
	print(v)
	v = Vector(5)
	print(v)
	v = Vector(15, 19)
	print(v)
	w = v + 2
	print(w + v)
	print(w - v)
	v = Vector(5) * 2
	print(v)
	print(v / 2)
	x = Vector([3.0, -1.0])
	v = Vector([1.0, 2.0])
	print(x * v)

	print(repr(v))