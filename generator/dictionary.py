from typing import Set
import functools
import operator


class DictItem():
    __slots__ = ('text', 'props')
    def __init__(self, text: str, **kwargs):
        self.text = text
        self.props = kwargs

    def of(self, name):
        return self.props[name]

class Dictionary():
    __slots__ = ('tokens', 'children_of')
    key_t = str

    def __init__(self):
        self.tokens = dict()
        self.children_of = dict()

    def _add_key(self, key: key_t):
        if key not in self.tokens:
            self.tokens[key] = set()

    def add_token(self, key: key_t, value: DictItem):
        self._add_key(key)
        self.tokens[key].add(value)

    def set_child_relation(self, parent: key_t, child: key_t):
        if parent not in self.children_of:
            self.children_of[parent] = set()
        self.children_of[parent].add(child)

    def resolve_keys(self, key) -> Set[key_t]:
        q = [key]
        res = set()

        while q:
            c = q[0]
            q.pop(0)
            res.add(c)

            if c in self.children_of:
                for k in self.children_of[c]:
                    if k not in res:
                        q.append(k)

        return res

    def resolve_items(self, key) -> Set[DictItem]:
        keys = self.resolve_keys(key)

        return functools.reduce(
                operator.or_,
                map(lambda k: self.tokens[k] if k in self.tokens else set(), keys),
                set())

# alias
DictToken = DictItem

