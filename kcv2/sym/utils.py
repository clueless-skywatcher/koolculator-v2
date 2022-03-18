from abc import abstractmethod, ABC
        
class UtilityFunction(ABC):
    def __new__(cls, *args, **kwargs) -> None:
        __eval = cls.eval(*args, **kwargs)
        return __eval

    @staticmethod
    def eval(*args, **kwargs):
        pass
