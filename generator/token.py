from typing import Union
import copy
import numbers
import random
from . import korutil
from .dictionary import DictItem


class Token():
    def __init__(self, token_id):
        self._token_id = token_id

    def __str__(self):
        return self._token_id

    # token context
    @property
    def token(self) -> str:
        return '#{}'.format(self._token_id)

    # expression context
    @property
    def value(self):
        raise NotImplementedError()

    # string context
    @property
    def text(self):
        raise NotImplementedError()


# one-time execution
class DelayedExecution():
    def __init__(self, obj):
        self.fns = []
        self.obj = obj

    def apply(self, fn):
        self.fns.append(fn)

    def __call__(self):
        if self.fns is not None:
            for fn in self.fns:
                self.obj = fn(self.obj)
            self.fns = None
        return self.obj


class TextToken(Token):
    def __init__(self, token_id, text: Union[str, DictItem]):
        super().__init__(token_id)
        self._text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.value

    @property
    def text(self):
        if type(self._text) == DictItem:
            return self._text.text
        else:
            return self._text

    @property
    def value(self):
        return repr(self.text)

    def of(self, name):
        if type(self.value) == DictItem:
            return self.value.of(name)
        else:
            raise NotImplementedError()


class NumberToken(Token):
    def __init__(self, token_id, value):
        super().__init__(token_id)
        self._value = value
        self._unit = None

        self.sgn = 1

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.value

    @property
    def text(self):
        unit = ''
        if self._unit is not None:
            unit = self._unit
        if self.sgn == 1:
            return str(self._value) + unit
        elif self.sgn == -1:
            return str(-self._value) + unit
        else:
            return '0' + unit

    @property
    def value(self):
        return self._value * self.sgn

    @property
    def token(self):
        tkn = super().token
        if self.sgn == 1:
            return tkn
        elif self.sgn == 0:
            return '0' + tkn
        elif self.sgn == -1:
            return '-' + tkn

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value

    @unit.deleter
    def unit(self):
        self._unit = None

    def __neg__(self):
        x = copy.deepcopy(self)
        x.sgn *= -1
        return x

    def to_kor(self):
        return korutil.num2kor(self.value)
    def to_korcnt(self):
        return korutil.num2korcnt(self.value)
    def to_korunit(self):
        if self._unit is None:
            return korutil.num2korunit(self.value)
        else:
            return korutil.num2korunit(self.value) + ' ' + self._unit
    def to_korord(self):
        return korutil.num2korord(self.value)



class TokenPool():
    def __init__(self):
        self._token_id = 0

    def _get_token_id(self):
        self._token_id += 1
        return self._token_id

    def new(self, obj, *, type=None):
        if isinstance(obj, Token):
            return obj
        elif type is not None:
            return type(self._get_token_id(), obj)
        elif isinstance(obj, str) or isinstance(obj, DictItem):
            return TextToken(self._get_token_id(), obj)
        elif isinstance(obj, numbers.Number):
            return NumberToken(self._get_token_id(), obj)
        else:
            raise RuntimeError("Unknown type: {}".format(repr(obj)))

    def sample(self, lst, k=1):
        return list(map(self.new, random.sample(lst, k=k)))

    def randint(self, start, end=None):
        return self.new(random.randint(start, end))
