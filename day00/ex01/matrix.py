#!/usr/bin/python3

class Matrix:
	def __init__(self, *args):
		try:
			self.data = []
			if len(args) == 1 and type(args[0]) == list:
				for i in range(len(args[0])):
					self.data.append([])
				n = 0
				for i in args[0]:
					if (type(i) != list):
						raise ValueError
					for j in i:
						self.data[n].append(float(j))
					n += 1
				self.shape = (len(args[0]), len(self.data[0]))
			elif len(args) == 1 and type(args[0]) == tuple and len(args[0]) == 2:
				self.shape = args[0]
				for i in range(self.shape[0]):
					self.data.append([])
					for j in range(self.shape[1]):
						self.data[i].append(0.0)
			elif len(args) == 2 and type(args[0]) == list and type(args[1]) == tuple and len(args[0]) == args[1][0]:
				self.data = []
				for i in range(len(args[0])):
					self.data.append([])
				n = 0
				for i in args[0]:
					if (type(i) != list):
						raise ValueError
					for j in i:
						self.data[n].append(float(j))
					n += 1
				self.shape = args[1]
			else:
				raise ValueError
		except ValueError:
			print("Initialization error")
	
	def __str__(self):
		txt = "Matrix = "
		txt += "".join(str(self.data))
		return txt

	def __repr__(self):
		txt = "Matrix ="
		txt += " { data: " + "".join(str(self.data))
		txt += " , shape: " + "".join(str(self.shape)) + " }"
		return txt

	def __add__(self, val):
		try:
			mres = []
			if (self.shape != val.shape):
				raise IndexError
			for i in range(self.shape[0]):
				mres.append([])
			for i in range(self.shape[0]):
				for j in range(self.shape[1]):
					mres[i].append(self.data[i][j] + val.data[i][j])
			return(Matrix(mres))
		except IndexError:
			print("Error: Matrices are not same dimension")

	def __radd__(self, val):
		return val + self

	def __sub__(self, val):
		try:
			mres = []
			if (self.shape != val.shape):
				raise IndexError
			for i in range(self.shape[0]):
				mres.append([])
			for i in range(self.shape[0]):
				for j in range(self.shape[1]):
					mres[i].append(self.data[i][j] - val.data[i][j])
			return(Matrix(mres))
		except IndexError:
			print("Error: Matrices are not same dimension")
	
	def __rsub__(self, val):
		return val - self
	
	def __truediv__(self, val):
		try:
			mres = []
			for i in range(self.shape[0]):
				mres.append([])
			for i in range(self.shape[0]):
				for j in range(self.shape[1]):
					mres[i].append(self.data[i][j] / val)
			return Matrix(mres)
		except ZeroDivisionError:
			print("Error: non divisible by 0")
		except TypeError:
			print("Error: invalid type")

	def __rtruediv__(self, val):
		try:
			mres = []
			for i in range(self.shape[0]):
				mres.append([])
			for i in range(self.shape[0]):
				for j in range(self.shape[1]):
					mres[i].append(self.data[i][j] / val)
			return Matrix(mres)
		except ZeroDivisionError:
			print("Error: non divisible by 0")
		except TypeError:
			print("Error: invalid type")

	def __mul__(self, val):
		try:
			mres = []
			if (self.shape[1] != val.shape[0]):
				raise IndexError
			for i in range(self.shape[0]):
				mres.append([])
			for i in range(self.shape[0]):
				for j in range(val.shape[1]):
					res = 0.0
					for k in range(val.shape[0]):
						res += (self.data[i][k] * val.data[k][j])
					mres[i].append(res)
			print(mres)
			return Matrix(mres)
		except IndexError:
			print("Error: Matrices are not same dimension")

	def __rmul__(self, val):
		return (val * self)
