#!/usr/bin/python3

class Vector:
    def __init__(self, *args):
        try:
            self.values = []
            if len(args) == 1 and type(args[0]) == list and len(args[0]) > 0:
                for i in args[0]:
                    self.values.append(float(i))
                self.size = len(self.values)
            elif len(args) == 1 and type(args[0]) == int and args[0] > 0:
                for i in range(args[0]):
                    self.values.append(float(i))
                self.size = len(self.values)
            elif len(args) == 2 and type(args[0]) == int and type(args[1]) == int and args[1] > args[0]:
                for i in range(args[0], args[1]):
                    self.values.append(float(i))
                self.size = len(self.values)
            else:
                raise ValueError
        except ValueError:
            print("Error: Invalid vector initialization")

    def __str__(self):
        txt = "Vector "
        txt += "".join(str(self.values))
        return txt

    def __repr__(self):
        txt = "{ values: " + str(self.values) \
            + ", size: " + str(self.size) + "}"
        return txt

    def __add__(self, val):
        try:
            values = []
            if type(val) == int:
                for i in self.values:
                    values.append(i + val)
                return Vector(values)
            if len(self.values) != len(val.values):
                raise IndexError
            for i in range(self.size):
                values.append(self.values[i] + val.values[i])
            return Vector(values)
        except IndexError:
            print("Error vectors are not the same size")
        
    def __radd__(self, val):
        return (val + self)

    def __sub__(self, val):
        try:
            values = []
            if type(val) == int:
                for i in self.values:
                    values.append(i - val)
                return Vector(values)
            if len(self.values) != len(val.values):
                raise IndexError
            for i in range(self.size):
                values.append(self.values[i] - val.values[i])
            return Vector(values)
        except IndexError:
            print("Error: vectors are not the same size")

    def __rsub__(self, val):
        return (val - self)

    def __truediv__(self, val):
        try:
            values = []
            for i in self.values:
                values.append(i / val)
            return Vector(values)
        except ZeroDivisionError:
            print("Error: non divisible by 0")
        except TypeError:
            print("Error: invalid type")

    def __rtruediv__(self, val):
        try:
            values = []
            for i in self.values:
                values.append(val / i)
            return Vector(values)
        except ZeroDivisionError:
            print("Error: non divisible by 0")
        except TypeError:
            print("Error: invalid type")

    def __mul__(self, val):
        try:
            if type(val) == Vector:
                if self.size != val.size:
                    raise IndexError
                result = 0
                for i in range(self.size):
                    result += (self.values[i] * val.values[i])
                return result
            values = []
            for i in self.values:
                values.append(i * val)
            return Vector(values)
        except IndexError:
            print("Error: mismatching vector lengths")
        except TypeError:
            print("Error: multiplying wrong type")

    def __rmul__(self, val):
        return (self * val)
