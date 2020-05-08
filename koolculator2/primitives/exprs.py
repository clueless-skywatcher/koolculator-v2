class Expression:
    def __init__(self, *args):
        is_assoc = False # Is the expression associative?
        is_commut = False # Is the expression commutative?

        self._args = args

    @property
    def class_name(self):
        return self.__class__.__name__

    def get_args(self):
        return self._args

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return False
        elif self.class_name != other.class_name:
            return False
        else:
            for arg1, arg2 in zip(self.get_args(), other.get_args()):
                if arg1 != arg2:
                    return False

            return True

    def __str__(self):
        pass