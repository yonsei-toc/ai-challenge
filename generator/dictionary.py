from typing import List, Set
import functools
import operator
import json


class Namespace():
    pass


class DictItem():
    __slots__ = ('text', 'props', 'tags')

    def __init__(self, text: str, tags=[], **kwargs):
        self.text = text
        self.tags = set(tags)
        self.props = kwargs

    def of(self, name):
        return self.props[name]


class Dictionary():
    __slots__ = ('tokens', 'children_of')
    key_t = str

    def __init__(self):
        self.tokens = list()
        self.children_of = dict()

    def export(self, alias=None, **kwargs):
        data = {
            'alias': { k : v for k, v in alias.__dict__.items() if alias is not None },
            'dictionary': {
                'children_of': {
                    k: list(v) for k, v in self.children_of.items()
                },
                'tokens': [
                    {
                        'text': x.text,
                        'tags': list(x.tags),
                        'props': x.props
                    } for x in self.tokens
                ]
            }
        }

        return json.dumps(data, **kwargs)

    @staticmethod
    def load(s):
        data = json.loads(s)

        d = Dictionary()
        a = Namespace()

        d.children_of = { k: set(v) for k, v in data['dictionary']['children_of'].items() }
        d.tokens = list( DictItem(x['text'], x['tags'], **x['props']) for x in data['dictionary']['tokens'] )
        a.__dict__ = data['alias']

        return d, a

    def add_token(self, key: key_t, value: DictItem):
        return self.add_token_tag(value, key)

    def add_token_tag(self, value: DictItem, *tags: List[str]):
        value.tags |= set(tags)
        self.tokens.append(value)

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
                for k in self.children_of[c] - res:
                    q.append(k)

        return res

    def resolve_items(self, *keys) -> Set[DictItem]:
        # retrieve common keys
        resolved_keys = functools.reduce(
                operator.and_,
                map(self.resolve_keys, keys))

        return set(filter(lambda x: resolved_keys & x.tags, self.tokens))


# alias
DictToken = DictItem

