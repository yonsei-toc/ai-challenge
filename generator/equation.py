from typing import Callable, Iterable, Optional

class Equations():
    id_t = str

    def __init__(self):
        self.equations = dict()

    # decorator
    def register(self, func: Callable[..., str], id: Optional[id_t] = None, *, variable = None):
        if id is None:
            id = func.__name__

        if id in self.equations:
            raise RuntimeError("Duplicated equation ID: {id}.".format(id=id))

        self.equations[id] = func

        return func

    def get(self, key: id_t) -> Callable[..., str]:
        return self.equations[key]

    def __iter__(self) -> Iterable[Callable[..., str]]:
        return iter(self.equations.values())


equations = Equations()
