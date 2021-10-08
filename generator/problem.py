from .dictselector import DictionarySelector
from .token import TokenPool
from typing import Callable

class Problems():
    def __init__(self):
        self.problems = []

    # decorator
    def register(self, fn: Callable[[DictionarySelector, TokenPool], dict]):
        self.problems.append(fn)

    def __iter__(self):
        return iter(self.problems)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, key):
        return self.problems[key]


class EqnRef():
    def __init__(self, eqnid, *args):
        self._eqnid = eqnid
        self._args = args

    @property
    def eqnid(self): return self._eqnid

    @property
    def args(self): return self._args

problems = Problems()

# @problems.register
# def prob01(selector, tokenpool):
#     # get dict item
#     item = selector.select('item')

#     # ctor tokens
#     item = tokenpool.new(item)

#     count1 = tokenpool.randint(1, 100)
#     count2 = tokenpool.randint(1, count1)

#     return template.build(
#             body = '{item}#{이} {count1.add_unit(item.unit)}#{가} 있습니다.',
#             question = '{item}#{은} 모두 몇 {item.unit}인가요?',
#             equation = EqnRef('eqn1', count1),

#             local = token_map(
#                 item=item,
#                 count1=count1
#             ))
