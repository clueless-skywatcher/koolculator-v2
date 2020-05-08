from koolculator2.primitives import exprs

class Atomic(exprs.Expression):
    def __init__(self):
        pass

class Numeral(Atomic):
    pass

class RealNum(Numeral):
    def __init__(self, val):
        self.val = val
        super().__init__()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.val == other.val

    def __str__(self):
        return self.val

class Integer(RealNum):
    def __init__(self, val):
        self.val = val
        super().__init__(val)

class Fractional():
    def __init__(self, num, denom):
        self.num = num
        self.denom = denom

    def __str__(self):
        return f"({self.num}/{self.denom})"