from typing import List, Set
import functools
import operator
import json


class NamingProcessor:
    _target_tags = {'entity.name', 'entity.subject'}

    def __init__(self, dict_file='dict.json'):
        with open(dict_file, 'r', encoding='utf-8-sig') as f:
            dictionary, clskey = Dictionary.load(f.read())
        self.tokens = [str(t) for t in dictionary.tokens if (self._target_tags & t.tags) and len(str(t)) > 1]
        self.tokens.sort(key=len, reverse=True)

    def replace_token(self, s):
        i = 1
        names = {}
        for token in self.tokens:
            if token in s:
                names[f"[NAME{i}]"] = token
                s = s.replace(token, f"[NAME{i}]")
                i += 1
        return s, names

    def replace_batch(self, batch):
        replacements, names = [], []
        for q in batch:
            replaced, nums = self.replace_token(q)
            replacements.append(replaced), names.append(nums)
        return replacements, names


class Namespace():
    pass


class DictItem():
    __slots__ = ('text', 'props', 'tags')

    def __init__(self, text: str, tags=[], **kwargs):
        self.text = text
        self.tags = set(tags)
        self.props = kwargs

    def __str__(self):
        return self.text

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
            'alias': {k: v for k, v in alias.__dict__.items() if alias is not None},
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

        d.children_of = {k: set(v) for k, v in data['dictionary']['children_of'].items()}
        d.tokens = list(DictItem(x['text'], x['tags'], **x['props']) for x in data['dictionary']['tokens'])
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
            c = q.pop(0)
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


if __name__ == '__main__':
    np = NamingProcessor()
    print(np.replace_token("은지는 윤기보다 많다. [NUM]명 중 가장 가장 많은 사람을 구하시오. 태형은 은지보다 많다. 유정은 윤기보다 적고 태형은 유정보다 적다. 석진은 영수보다 크고 영수는 남준보다 크다."))
