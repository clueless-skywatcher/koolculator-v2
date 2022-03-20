from abc import ABC, abstractmethod
from copy import copy
import functools
import math

from kcv2.sym.utils import UtilityFunction

def koolculatorize(expr):
    if isinstance(expr, Expression):
        return expr
    elif isinstance(expr, (int, float)):
        return Numeric(expr)
    else:
        raise ValueError(f"Cannot convert {expr} to Expression")

def _is_negative(expr):
    _, numeric = _separate_numeric_from_mul(expr)
    return numeric.n < 0

def _make_positive(expr):
    if _is_negative(expr):
        return expr * Numeric(-1)
    return expr

def _separate_numeric_from_mul(mul):
    if isinstance(mul, Mul):
        numerics = []
        vars = []
        prod = lambda x, y: x * y
        for exp in mul.exps:
            if isinstance(koolculatorize(exp), Numeric):
                numerics.append(koolculatorize(exp))
            else:
                vars.append(koolculatorize(exp))
        
        if len(numerics) > 0:
            numeric_prod = functools.reduce(prod, numerics)
        else:
            numeric_prod = 1
        if len(vars) == 1:
            return vars[0], koolculatorize(numeric_prod)
        return Mul(*vars), koolculatorize(numeric_prod)
    elif isinstance(mul, (Var, Pow)):
        return mul, Numeric(1)
    else:
        return None, Numeric(1)

def find_distinct_variables(expr):
    if isinstance(expr, Var):
        return set(expr.symbol)
    elif isinstance(expr, Numeric):
        return set()
    elif isinstance(expr, (Sum, Mul)):
        return set().union(*[find_distinct_variables(exp) for exp in expr.exps])
    elif isinstance(expr, Pow):
        return find_distinct_variables(expr.base).union(find_distinct_variables(expr.exp))
    elif isinstance(expr, ApplyFunction):
        return find_distinct_variables(expr.arg)
    else:
        raise TypeError("Not a valid expression")

class Expression(ABC):
    @abstractmethod
    def evaluate(self, **bindings):
        pass

    def __add__(self, other):
        if koolculatorize(other) == Numeric(0):
            return self
        return Sum(self, koolculatorize(other))

    def __radd__(self, other):
        return self.__add__(koolculatorize(other))

    def __sub__(self, other):
        return Sum(self, -koolculatorize(other))

    def __mul__(self, other):
        if koolculatorize(other) == Numeric(0):
            return Numeric(0)
        if koolculatorize(other) == Numeric(1):
            return self
        return Mul(self, koolculatorize(other))

    def __neg__(self):
        return Numeric(-1) * self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Div(self, koolculatorize(other))

    def __pow__(self, other):
        return Pow(self, koolculatorize(other))

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def srepr_display(self):
        pass

    def __str__(self) -> str:
        return self.display()

    def __repr__(self) -> str:
        return self.display()

class Numeric(Expression):
    def __init__(self, n) -> None:
        self.n = n
    
    def evaluate(self):
        return self.n
    
    def __eq__(self, __o: object) -> bool:
        if isinstance(koolculatorize(__o), Numeric):
            return self.n == __o.n
        return NotImplemented

    def display(self):
        return str(self.n)

    def srepr_display(self):
        return f"Numeric({self.n})"

    def __hash__(self) -> int:
        return hash(self.n)

    def __add__(self, other):
        other = koolculatorize(other)
        if isinstance(other, Numeric):
            return Numeric(self.n + other.n)
        return Sum(self, other)

    def __mul__(self, other):
        other = koolculatorize(other)
        if isinstance(other, Numeric):
            return Numeric(self.n * other.n)
        return Mul(self, koolculatorize(other))

    def __pow__(self, other):
        other = koolculatorize(other)
        if isinstance(other, Numeric):
            return Numeric(self.n ** other.n)
        return Pow(self, koolculatorize(other))

    def __rpow__(self, other):
        other = koolculatorize(other)
        if isinstance(other, Numeric):
            return Numeric(other.n ** self.n)
        return Pow(other, self)

class RationalNumber(Numeric):
    pass

class IrrationalNumber(Numeric):
    pass
        
class Var(Expression):
    def __init__(self, symbol) -> None:
        self.symbol = symbol

    def evaluate(self, **bindings):
        try:
            return bindings[self.symbol]
        except:
            raise KeyError(f"Did not find {self.symbol} in bindings")        

    def __eq__(self, other: object) -> bool:
        if isinstance(koolculatorize(other), Var):
            return self.symbol == other.symbol
        else:
            return False

    def srepr_display(self):
        return f"Var({self.symbol})"

    def display(self):
        return self.symbol

    def __hash__(self) -> int:
        return hash(self.symbol)

class Sum(Expression):    
    def __init__(self, *exps) -> None:
        self.exps = exps
        self.exps = self._flatten_sums(self.exps)
        self.exps = self._arrange_exps(self.exps)
        self.exps = self._remove_zeros(self.exps)

    def _remove_zeros(self, exps):
        return [exp for exp in exps if koolculatorize(exp) != Numeric(0)]

    @staticmethod
    def _flatten_sums(exps):
        new_exps = []
        for exp in exps:
            if isinstance(exp, Sum):
                new_exps.extend(Sum._flatten_sums(exp.exps))
            else:
                new_exps.append(exp)
        return new_exps

    def srepr_display(self):
        return f"Add({', '.join([exp.srepr_display() for exp in self.exps])})"

    @staticmethod
    def _arrange_exps(exps):
        numerics = [x for x in exps if isinstance(x, Numeric)]
        
        mul_dict = {}
        for exp in exps:
            if not isinstance(koolculatorize(exp), Numeric):
                mul, _ = _separate_numeric_from_mul(exp)
                mul_dict[mul] = mul_dict.get(mul, 0)        
        numeric_sum = sum(numerics)       

        for exp in exps:
            if not isinstance(exp, Numeric):
                mul, num = _separate_numeric_from_mul(exp)
                if mul is not None:
                    mul_dict[mul] += koolculatorize(num)
        final_list = [numeric_sum, *[mul * num for mul, num in mul_dict.items() if koolculatorize(num) != Numeric(0)]]
        if len(final_list) == 0:
            final_list = [Numeric(0)]
        return final_list

    def evaluate(self, **bindings):
        return sum([exp.evaluate(**bindings) for exp in self.exps])

    def alternate_display(self):
        first_exp = self.exps[0]
        display_str = f"({str(first_exp)})" if isinstance(first_exp, Sum) else str(first_exp)
        for i in range(1, len(self.exps)):
            if _is_negative(self.exps[i]):
                display_str += f" - {str(_make_positive(self.exps[i]))}"
            else:
                display_str += f" + {str(self.exps[i])}"
        return display_str

    def display(self):
        if len(self.exps) > 0:
            return self.alternate_display()
        else:
            return "0"
        
    def __hash__(self) -> int:
        return hash(tuple(self.exps))

    def __eq__(self, __o: object):
        if isinstance(__o, Sum):
            return set(self.exps) == set(__o.exps)
        return False

class Mul(Expression):
    def __init__(self, *exps) -> None:
        self.exps = exps
        self.exps = self._flatten_muls(self.exps)
        self.exps = self._arrange_exps(self.exps)
        self.exps = self._remove_ones(self.exps)
        self.exps = self._evaluate_zero(self.exps)

    @staticmethod
    def _evaluate_zero(exps):
        return [Numeric(0)] if Numeric(0) in [koolculatorize(exp) for exp in exps] else exps

    @staticmethod
    def _remove_ones(exps):
        return [exp for exp in exps if koolculatorize(exp) != Numeric(1)]

    def srepr_display(self):
        return f"Mul({', '.join([koolculatorize(exp).srepr_display() for exp in self.exps])})"

    @staticmethod
    def _arrange_exps(exps):
        exps = [koolculatorize(exp) for exp in exps]
        numerics = [x for x in exps if isinstance(x, Numeric)]
        vars = [x for x in exps if isinstance(x, Var)]
        sums = [x for x in exps if isinstance(x, Sum)]
        pows = [x for x in exps if isinstance(x, Pow)]
        others = [x for x in exps if not isinstance(x, (Numeric, Var, Sum, Mul, Pow))]

        prod = lambda x, y: x * y

        end_list = []

        if len(numerics) > 0:
            numeric_prod = functools.reduce(prod, numerics)
            end_list.append(numeric_prod)

        if len(vars) > 0:
            pow_dict = {x: 0 for x in vars}
            for var in vars:
                pow_dict[var] += 1
            
            var_list = []
            for var, pow in pow_dict.items():
                pow = koolculatorize(pow)
                if pow == Numeric(1):
                    var_list.append(var)
                else:
                    var_list.append(Pow(var, pow))

            def sort_var_key(var):
                if isinstance(var, Var):
                    return var.symbol
                elif isinstance(var, Pow) and isinstance(var.base, Var):
                    return var.base.symbol
                return NotImplemented

            var_list = sorted(var_list, key = sort_var_key)
            end_list.extend(var_list)

        end_list.extend(sums)
        end_list.extend(others)
        end_list.extend(pows)
        return end_list

    @staticmethod
    def _flatten_muls(exps):
        new_exps = []
        for exp in exps:
            if isinstance(exp, Mul):
                new_exps.extend(Mul._flatten_muls(exp.exps))
            elif isinstance(exp, Numeric):
                new_exps.insert(0, exp)
            else:
                new_exps.append(exp)

        return new_exps        
    
    def evaluate(self, **bindings):
        _mul_func = lambda x, y: x * y
        return functools.reduce(_mul_func, [exp.evaluate(**bindings) for exp in self.exps])

    def display(self):
        str_rep = "*".join([f"({str(exp)})" if isinstance(exp, (Mul, Sum)) else str(exp) for exp in self.exps])
        if Numeric(-1) in self.exps:
            positive = _make_positive(self)
            str_rep = "*".join([f"({str(exp)})" if isinstance(exp, (Mul, Sum)) else str(exp) for exp in positive.exps])
            str_rep = f"-{str_rep}"
        return str_rep

    def __hash__(self) -> int:
        return hash(tuple(self.exps))

    def __eq__(self, __o: object):
        if isinstance(__o, Mul):
            return set(self.exps) == set(__o.exps)
        return False

    def __pow__(self, other):
        other = koolculatorize(other)
        return Mul(*[exp ** other for exp in self.exps])

class Div(Expression):
    def __init__(self, num, denom) -> None:
        self.num = num
        self.denom = denom

    def evaluate(self, **bindings):
        return self.num.evaluate(**bindings) / self.denom.evaluate(**bindings)

    def __hash__(self) -> int:
        return hash(tuple(self.num, self.denom))

class Negative(Expression):
    def __init__(self, exp) -> None:
        self.exp = exp

    def evaluate(self, **bindings):
        return -self.exp.evaluate(**bindings)


class Pow(Expression):
    def __init__(self, base, exp) -> None:
        self.base = base
        self.exp = exp

    def srepr_display(self):
        return f"Pow({self.base}, {self.exp})"

    def evaluate(self, **bindings):
        return self.base.evaluate(**bindings) ** self.exp.evaluate(**bindings)

    def __hash__(self) -> int:
        return hash(tuple([self.base, self.exp]))

    def __mul__(self, other):
        if isinstance(other, Pow):
            if self.base == other.base:
                return Pow(self.base, self.exp + other.exp)
        return Mul(self, other)

    def __pow__(self, other):
        return Pow(self.base, koolculatorize(self.exp) * koolculatorize(other))

    def display(self):
        base_rep = self.base.display()
        exp_rep = self.exp.display()
        if isinstance(self.base, Sum):
            base_rep = f"({base_rep})"
        if isinstance(self.exp, (Sum, Mul)):
            exp_rep = f"({exp_rep})"

        return f"{base_rep}^{exp_rep}"

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Pow):
            return self.base == __o.base and self.exp == __o.exp
        return False

_FUNC_BINDINGS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "ln": math.log
}

class ApplyFunction(Expression):
    def __init__(self, func, arg) -> None:
        self.func = func
        self.arg = arg

    def evaluate(self, **bindings):
        return _FUNC_BINDINGS[self.func](self.arg.evaluate(**bindings))

    def display(self):
        return f"{self.__class__.__name__}({str(self.arg)})"

    def srepr_display(self):
        return f"{self.__class__.__name__}({self.arg.srepr_display()})"

class NumberOfOperands(UtilityFunction):
    def eval(expr):
        if isinstance(expr, (Sum, Mul)):
            return len(expr.exps)
        elif isinstance(expr, (Div)):
            return 2
        elif isinstance(expr, (Var, Numeric)):
            return 1

class NthOperand(UtilityFunction):
    def eval(expr, i):
        try:
            if isinstance(expr, (Sum, Mul)):
                return expr.exps[i]
            elif isinstance(expr, Div):
                return [expr.num, expr.denom][i]
            elif isinstance(expr, (Var, Numeric)):
                if i == 0:
                    return expr
                else:
                    raise ValueError(f"Cannot find {i}th operand of expression: {expr}")
        except:
            raise ValueError(f"Cannot find {i}th operand of expression: {expr}")

class FreeOf(UtilityFunction):
    def eval(expr, free_of):
        free_of = koolculatorize(free_of)
        if expr == free_of:
            return False
        if isinstance(expr, Var):
            return expr != free_of
        elif isinstance(expr, Numeric):
            return False 
        if isinstance(expr, (Sum, Mul)):
            return all([FreeOf.eval(exp, free_of) for exp in expr.exps])
        if isinstance(expr, Div):
            return FreeOf.eval(expr.num, free_of) and FreeOf.eval(expr.denom, free_of)

        raise NotImplementedError

class ObjectRepresentation(UtilityFunction):
    def eval(expr):
        return expr.srepr_display()
        
