from . import korutil
import random


class _Templates():
    def __init__(self):
        self.fns = set()

    def register(self, fn):
        self.fns.add(fn)
        return fn


templates = _Templates()
register = templates.register


class EntityToken():
    def __init__(self, key, value, **kwargs):
        self.key = key
        self.props = kwargs
        self.__value = value

    @property
    def value(self):
        return self.__value

    @property
    def unit(self):
        return self.of('unit')

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    # replicate str
    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def of(self, name):
        return self.props[name]


class NumericToken():
    def __init__(self, value):
        self.__value = value

    @property
    def value(self):
        return self.__value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def to_kor(self):
        return korutil.num2kor(self.value)

    def to_korcnt(self):
        return korutil.num2korcnt(self.value)

    def to_korunit(self):
        return korutil.num2korunit(self.value)

    def to_korord(self):
        return korutil.num2korord(self.value)


class TokenRegistry():
    def __init__(self):
        self.tokens = dict()
        self.hier = dict()

    def add_key(self, key):
        if key not in self.tokens:
            self.tokens[key] = set()

    def add_token(self, key, value, **kwargs):
        if key not in self.tokens:
            self.add_key(key)

        self.tokens[key].add(EntityToken(key, value, **kwargs))

    def add_hierarchy(self, key, subkey):
        if key not in self.hier:
            self.hier[key] = set()
        self.hier[key].add(subkey)


class TokenSelector():
    def __init__(self, registry):
        self.registry = registry
        self.drawn = set()

    def _get_all_subkeys(self, key):
        keys = set()
        q = [key]

        while q:
            k = q[0]
            keys.add(k)
            q.pop(0)

            if key in self.registry.hier:
                q += list(self.registry.hier[key] - keys)
        return keys

    def _get_subkey(self, key):
        keys = self._get_all_subkeys(key)

        tokens = set()

        for key in keys:
            if key in self.registry.tokens:
                tokens |= self.registry.tokens[key]

        return tokens

    def get(self, key):
        tokens = self._get_subkey(key)

        val = random.choice(list(tokens - self.drawn))
        self.drawn.add(val)
        return val

    def randint(self, start=0, end=None):
        if end is None:
            end = start
            start = 0

        if isinstance(start, NumericToken):
            start = start.value
        if isinstance(end, NumericToken):
            end = end.value

        return NumericToken(random.randint(start, end))

    def randreal(self, start=0, end=None, ndigits=2):
        if end is None:
            end = start
            start = 0

        if isinstance(start, NumericToken):
            start = start.value
        if isinstance(end, NumericToken):
            end = end.value

        return NumericToken(round(random.uniform(start, end), ndigits=ndigits))

    def append_connection(self, target, type):
        return korutil.append_connection(str(target), type)
