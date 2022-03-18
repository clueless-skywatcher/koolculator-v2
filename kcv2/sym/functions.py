import math

from kcv2.sym.expression import ApplyFunction

class Sin(ApplyFunction):
    def __init__(self, arg) -> None:
        super().__init__('sin', arg)

class Cos(ApplyFunction):
    def __init__(self, arg):
        super().__init__('cos', arg)

class Tan(ApplyFunction):
    def __init__(self, arg) -> None:
        super().__init__('tan', arg)

class Sqrt(ApplyFunction):
    def __init__(self, arg):
        super().__init__('sqrt', arg)

class Ln(ApplyFunction):
    def __init__(self, arg):
        super().__init__('ln', arg)