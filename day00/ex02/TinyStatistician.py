import math

class TinyStatistician():
	def mean(self, x):
		if len(x) < 1:
			return None
		return (float(sum(x)) / len(x))
	
	def median(self, x):
		if len(x) < 1:
			return None
		n = len(x)
		x.sort()
		if n % 2 == 0:
			median1 = x[ n // 2]
			median2 = x[( n// 2) - 1]
			median = (median1 + median2) / 2.0
		else:
			median = float(x[n // 2])
		return median
	
	def quartiles(self, x, percentile):
		if len(x) < 1:
			return None
		x.sort()
		if len(x) % 2 == 0:
			median1 = x[:(len(x)//2)]
			median2 = x[len(x)//2:]
			if percentile == 25:
				return self.median(median1)
			elif percentile == 75:
				return self.median(median2)
		else:
			median1 = x[:(len(x)//2)]
			median2 = x[len(x)//2 + 1:]
			if percentile == 25:
				return self.median(median1)
			elif percentile == 75:
				return self.median(median2)
	
	def var(self, x):
		if len(x) < 1:
			return None
		m = self.mean(x)
		ret = 0.0
		for each in x:
			ret += float(each - m) ** 2
		return float(ret / len(x))

	def std(self, x):
		if len(x) < 1:
			return None
		return math.sqrt(self.var(x))



if __name__ == "__main__":
	ts = TinyStatistician()
	arr = [1, 42, 10, 300, 59]
	print(ts.median(arr))
	print(ts.quartiles(arr, 25))
	print(ts.quartiles(arr, 75))
	print(ts.var(arr))
	print(ts.std(arr))
