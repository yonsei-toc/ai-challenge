from typing import Callable, Iterable, Optional
import functools

class Equations():
    id_t = str

    def __init__(self):
        self.equations = dict()

    def _decorator(self, func: Callable[..., str], id: Optional[id_t] = None):
        if id is None:
            id = func.__name__

        if id in self.equations:
            raise RuntimeError("Duplicated equation ID: {id}.".format(id=id))

        self.equations[id] = func
        return func

    # decorator
    def register(self, id: Optional[id_t] = None, *, variable: Optional[str] = None):
        return functools.partial(self._decorator, id=id)

    def get(self, key: id_t) -> Callable[..., str]:
        return self.equations[key]

    def __iter__(self) -> Iterable[Callable[..., str]]:
        return iter(self.equations.values())


equations = Equations()
