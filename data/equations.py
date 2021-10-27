import math, itertools
from typing import Callable, Iterable, Optional
import random
import inspect


class Equations:
    key_t = str

    def __init__(self):
        self.equation_keys = {}
        self.equations = []

    def _decorator(self, func: Callable[..., Callable], key: Optional[key_t] = None):
        if key is None:
            key = func.__name__

        if key in self.equation_keys:
            raise RuntimeError("Duplicated equation ID: {}.".format(key))

        def _func_to_source(*args):
            result = func(*args)
            if not callable(result):
                return result
            codes = inspect.getsourcelines(result)[0][1:]

            indents = len(codes[0]) - len(codes[0].lstrip())
            codes = ['import math, itertools\n'] + [c[indents:] for c in codes] + ['print(ans)\n']

            _replaced = ''.join(codes)
            for i, arg in enumerate(args):
                _replaced = _replaced.replace(f'_args_[{i}]', repr(args[i]))
            _replaced = _replaced.replace('_args_', repr(list(args)))
            return _replaced

        self.equations.append(_func_to_source)
        self.equation_keys[key] = len(self.equations) - 1
        return _func_to_source

    # decorator
    def register(self, func, key=None):
        return self._decorator(func, key)

    def get(self, key: key_t) -> Callable[..., str]:
        return self.equations[self.equation_keys[key]]

    def get_id(self, key: key_t):
        return self.equation_keys[key]

    def __iter__(self) -> Iterable[Callable[..., str]]:
        return iter(self.equations)


equations = Equations()


# utils
def randreal(st, ed, *, ndigits=2):
    if ed is None:
        st, ed = 0, st

    if ndigits is None:
        return random.uniform(st, ed)
    else:
        return round(random.uniform(st, ed), ndigits=ndigits)


"""
equations
"""


# 0
@equations.register
def diff_perm(*_args_):
    """
    {1} 중에서 서로 다른 숫자 {0}개를 뽑아 -> 만들 수 있는 {0} 자리 수의 {2}

    :param
        {0} : 숫자
        {1} : 숫자 리스트
        {2} :   0 - 가장 큰 값 (max)
                1 - 가장 작은 값 (min)
                2 - 차이의 최대값 (max - min)
                3 - 경우의 수의 개수
    """

    if _args_[2] == 0:
        def _equation():
            ans = max(filter(lambda e: e >= 10 ** (_args_[0] - 1), map(lambda e: int(''.join(map(str, e))), itertools.permutations(_args_[1], _args_[0]))))
    elif _args_[2] == 1:
        def _equation():
            ans = min(filter(lambda e: e >= 10 ** (_args_[0] - 1), map(lambda e: int(''.join(map(str, e))), itertools.permutations(_args_[1], _args_[0]))))
    elif _args_[2] == 2:
        def _equation():
            ans = list(filter(lambda e: e >= 10 ** (_args_[0] - 1), map(lambda e: int(''.join(map(str, e))), itertools.permutations(_args_[1], _args_[0]))))
            ans = max(ans) - min(ans)
    else:
        def _equation():
            ans = math.perm(len(_args_[1]) - 1, _args_[0] - 1) * (len(_args_[1]) - 1) if 0 in _args_[1] else math.perm(len(_args_[1]), _args_[0])

    return _equation


# 1
@equations.register
def count_from_range(*_args_):
    """
    {0}부터 {1}까지 적었을 때 등장하는 {2}의 수

    :param
        {0}, {1}, {2} : 숫자
    """

    def _equation():
        ans = len(list(filter(lambda e: _args_[2] == e, ''.join(map(str, range(_args_[0], _args_[1] + 1))))))

    return _equation


# 2
@equations.register
def find_sum_from_range(*_args_):
    """ncm
    {0} ~ {1}까지의 수 중 {2}개의 수를 동시에 뽑아 그 합이 {3}이 되는 경우의 수
    """
    if _args_[0] == 0:
        def _equation():
            ans = len(list(filter(lambda e: e == _args_[3], map(sum, itertools.combinations(range(_args_[1]), _args_[2])))))
    else:
        def _equation():
            ans = len(list(filter(lambda e: e == _args_[3], map(sum, itertools.combinations(range(_args_[0], _args_[1]), _args_[2])))))

    return _equation


# 3
@equations.register
def wrong_multiply(*_args_):
    """
    {0} 자리수 X, Y중 한 수의 {1}의 자리 숫자 {2}를 {3}로 잘못 보고 계산하여
    {4}를 얻었다. 올바르게 계산한 값이 {5}일 때 X, Y 중 {6}
    :param
        {0} : 2, 3
        {1} : 1, 10, (100)
        {2}, {3} : 0 ~ 9
        {4}, {5} : 숫자
        {6} :   0 - 작은 수
                1 - 큰 수
                2 - X
                3 - Y

    """

    if _args_[6] == 0:
        def _equation():
            y = (_args_[4] - _args_[5]) // ((_args_[3] - _args_[2]) * _args_[1])
            ans = min(_args_[5] // y, y)
    elif _args_[6] == 1:
        def _equation():
            y = (_args_[4] - _args_[5]) // ((_args_[3] - _args_[2]) * _args_[1])
            ans = max(_args_[5] // y, y)
    elif _args_[6] == 2:
        def _equation():
            ans = _args_[5] // ((_args_[4] - _args_[5]) // ((_args_[3] - _args_[2]) * _args_[1]))
    else:
        def _equation():
            ans = (_args_[4] - _args_[5]) // ((_args_[3] - _args_[2]) * _args_[1])

    return _equation


# 4
# TODO : 이 부분은 autoregressive 로 구현해야 함
@equations.register
def order_by_comp(*_args_):
    """
    {2k} > {2k+1} 일 때, 가장 큰 사람은?
    :param
        {0~} : 토큰
    """

    def _equation():
        ans = _args_[0]

    return _equation


# 5
@equations.register
def half_sub(*_args_):
    """
    길이가 {0}인 철사로 직사각형을 만들었더니 철사가 남지도 모자라지도 않았습니다.
    직사각형의 가로 길이가 {1}일 때, 세로 길이는 몇 cm입니까?
    """

    def _equation():
        ans = _args_[0] // 2 - _args_[1]

    return _equation


# 6
@equations.register
def sum_num_sig(*_args_):
    def _equation():
        ans = sum(_args_)

    return _equation


# 7
@equations.register
def max_sub_min(*_args_):
    def _equation():
        ans = max(_args_[0]) - min(_args_[0])

    return _equation


# 8
@equations.register
def max_sub_min2(*_args_):
    def _equation():
        ans = max(_args_) - min(_args_)

    return _equation


# 9
@equations.register
def count_from_compare_pivot(*_args_):
    '''
    {2} 중에서 {1} 보다 {0} 수는 모두 몇 개입니까?
    :param: {1} - 숫자, {2} - 숫자 리스트
            {0} - 0 큰 / 1 작은 / 2 같거나 큰 / 3 같거나 작은
    '''
    # return variable is ALWAYS [ans]..
    if _args_[0] == 0:
        def _equation():
            ans = list(map(lambda x: x > _args_[1], _args_[2])).count(True)
    elif _args_[0] == 1:
        def _equation():
            ans = list(map(lambda x: x < _args_[1], _args_[2])).count(True)
    elif _args_[0] == 2:
        def _equation():
            ans = list(map(lambda x: x >= _args_[1], _args_[2])).count(True)
    else:
        def _equation():
            ans = list(map(lambda x: x <= _args_[1], _args_[2])).count(True)
    return _equation


# 10
@equations.register
def count_from_compare_pivot2(*_args_):
    '''
    {2} 중에서 {1} 보다 {0} 수는 모두 몇 개입니까?
    :param: {1} - 숫자, {2} - 숫자 리스트
            {0} - 0 큰 / 1 작은 / 2 같거나 큰 / 3 같거나 작은
    '''
    # return variable is ALWAYS [ans]..
    if _args_[0] == 0:
        def _equation():
            ans = list(map(lambda x: x > _args_[1], _args_[2:])).count(True)
    elif _args_[0] == 1:
        def _equation():
            ans = list(map(lambda x: x < _args_[1], _args_[2:])).count(True)
    elif _args_[0] == 2:
        def _equation():
            ans = list(map(lambda x: x >= _args_[1], _args_[2:])).count(True)
    else:
        def _equation():
            ans = list(map(lambda x: x <= _args_[1], _args_[2:])).count(True)
    return _equation


# 11
@equations.register
def avg_num_sig(*_args_):
    # return variable is ALWAYS [ans].
    def _equation():
        ans = sum(_args_) / len(_args_)

    return _equation


# 12
@equations.register
def mul_num(*_args_):
    # return variable is ALWAYS [ans].
    def _equation():
        ans = 1
        for i in _args_:
            ans = (ans * i) if i > 0 else ans

    return _equation


# 13
@equations.register
def max_num(*_args_):
    def _equation():
        ans = max(_args_)

    return _equation


# 14
@equations.register
def min_num(*_args_):
    def _equation():
        ans = min(_args_)

    return _equation


# 15
@equations.register
def max_nums(*_args_):
    def _equation():
        ans = max(_args_[0])

    return _equation


# 16
@equations.register
def min_nums(*_args_):
    def _equation():
        ans = min(_args_[0])

    return _equation


# 17
@equations.register
def avg_nums(*_args_):
    def _equation():
        ans = sum(_args_[0]) / len(_args_[0])

    return _equation


@equations.register
def eq_c5p2(n1, n2, n3, n4):
    # 상수 사용
    # range -> 한 자리 수 범위
    # cand이 여러 후보군이 될 수 있음에 따라 indexing
    return f'''cand = [(var1,var2) for var1 in range(10) for var2 in range(10) if {n1}*10 + var1 - var2*10 - {n2} == {n3}*10 + {n4} and var1!=var2]
(var1,var2)=cand[0]
ans = sum([var1,var2])'''


@equations.register
def eq_c5p3(n1, n2, n3, n4, n5):
    # 상수 사용
    # range -> 한 자리 수 범위
    # cand이 여러 후보군이 될 수 있음에 따라 indexing
    # 자릿수에 해당 -> 100,10 등을 곱함
    return f'''cand = [(var1,var2,var3,var4) for var1 in range(10) for var2 in range(1,10) for var3 in range(10) for var4 in range(1,10) if {n1}*100 + var1*10 + {n2} + var2*100 + {n3}*10 + var3 == var4*100 + {n4}*10 + {n5} and len({{var1,var2,var3,var4}})==4]
(var1,var2,var3,var4)=cand[0]
ans = var4'''


@equations.register
def eq_c5p4(divisor, quotient, remainder):
    return f'''{remainder} = max(range({divisor}))
ans = {divisor}*{quotient}+{remainder}'''


@equations.register
def eq_c5p5(n1, n4, n5, n6):
    # 상수 사용
    # 반올림 확인 위해 비교 -> 1000 곱함
    # 반올림 조건 확인 숫자 -> 5
    return f'''if {n4} == {n1}*1000:
    ans = len([var for var in range({n5},{n6}) if var < 5])
else:
    ans = len([var for var in range({n5},{n6}) if var >= 5])'''


@equations.register
def eq_c6p5(n1, n2, n3):
    return f'ans = {n1}*{n2}*{n3}'


@equations.register
def eq_c7p5(t1, t2, t3, t4, t5, index):
    return f'''sorted_ts=["{t1}","{t5}","{t2}","{t4}","{t3}"]; ans = sorted_ts[{index}-1]'''


@equations.register
def eq_c8p5(n1, n2):
    return f'''e2={n1} // (2 * ({n2}+1)); e1 = e2*{n2}; ans = e1'''


@equations.register
def half_odd(*args):
    # return variable is ALWAYS [ans].

    return 'ans = ({}//2) + 1'.format(args[0])


@equations.register
def get_deci(*args):
    # return variable is ALWAYS [ans].

    if args[1] == 2:
        return 'ans = round(float((100*{})/99),2)'.format(args[0])
    elif args[1] == 3:
        return 'ans = round(float((1000*{})/999),2)'.format(args[0])
    elif args[1] == 1:
        return 'ans = round(float((10*{})/9),2)'.format(args[0])
    else:
        return 'ans = round(float((1*{}))/1,2)'.format(args[0])


# factor
@equations.register
def factor(factor, result):
    return 'ans = {} // {}'.format(repr(result), repr(factor))


# total_before_split
@equations.register
def total_before_split(split_num, num, leftover=0):
    return 'ans = {} * {} + {}'.format(split_num, num, leftover)


# split_oops_split
@equations.register
def split_oops_split(split_num, num, split, leftover=0):
    return 'ans = ({} * {} + {})//{}'.format(split_num, num, leftover, split)


# select smallest from list
@equations.register
def select_small_from_three(*args):
    ans_str = '\n'.join(["val = [0,0,0]",
                         "val[0] = " + str(args[3]),
                         "val[1] = val[0]+" + str(args[4]),
                         "val[2] = val[1]-" + str(args[5]),
                         "max_i = 0",
                         "for i in range(1,3):",
                         "    if val[i]>val[max_i]:",
                         "        max_i = i",
                         "ans = [\"" + "\",\"".join([args[0], args[1], args[2]]) + "\"][max_i]"
                         ])
    return ans_str


# 수열
@equations.register
def num_sequence_with_diff(start, diff, length):
    return 'ans = [{}]'.format(','.join(map(str, [start + diff * i for i in range(length)])))


@equations.register
def halfOdd(*args):
    # return variable is ALWAYS [ans].

    return 'ans = ({}//2) + 1'.format(args[0])


@equations.register
def prob06_04(*args):
    # return variable is ALWAYS [ans].
    # args0 args1 args2
    return 'ans = math.ceil(((({}*{})+{}) / {}) - {})'.format(args[2], args[1], args[2], args[0], args[0])


@equations.register
def prob07_04(name0, name1, name2, name3, *args):
    # return variable is ALWAYS [ans].

    dict = {args[0]: name0.replace("'", ""), args[0] - args[1]: name1.replace("'", ""),
            args[2]: name2.replace("'", ""), args[0] + args[3]: name3.replace("'", "")}

    return 'ans = {}[max({}.keys())]'.format(str(dict), str(dict))


@equations.register
def prob04_03(over, *args):
    # return variable is ALWAYS [ans]..
    return 'ans = list(map(lambda x: x> {}, [{}])).count(True)'.format(int(over), ','.join(list(map(str, args))))


@equations.register
def prob08_04(*args):
    # return variable is ALWAYS [ans].
    # 'prob08_04', item1_k, item2_k, l_k, 한/두/세),

    dict = {"정삼각형": 3, "정사각형": 4, "정오각형": 5, "정팔각형": 8}
    dict_edge = {"한": 1, "두": 2, "세": 3}

    return 'ans = "%.2f" % float((({}/{})*{}) / {})'.format(args[2], dict_edge[args[3]], dict[args[0]], dict[args[1]])


@equations.register
def eqn02_1(lists, idx):
    return 'ans = {}[{}-1]'.format(lists, idx)


@equations.register
def eqn02_2(rank2, rank1):
    return 'ans = range(min({},{}), max({},{}))[0]+1'.format(rank2, rank1, rank2, rank1)


@equations.register
def eqn02_3(order1, n_people_btw):
    return 'ans = {}+{}+1'.format(order1, n_people_btw)


@equations.register
def eqn02_4(idx_left, idx_right, idx_front, idx_down):
    return 'ans = ({}+{}-1)*({}+{}-1)'.format(idx_left, idx_right, idx_front, idx_down)


@equations.register
def eqn02_5(idx_left_k, idx_right_k, n_floors, n_shelves_k):
    return 'ans = ({}+{}-1)*{}*{}'.format(idx_left_k, idx_right_k, n_floors, n_shelves_k)


@equations.register
def eqn07_2(n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, *variables):
    return f'ans = {variables}' + '[[{},{},{},int({}*{})].index(max({},{},{},int({}*{})))]'.format(n1, n1, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12,
                                                                                                   n13, n14, n15, n16)


@equations.register
def eqn08_2(length_k, n_k):
    return 'ans = {}*({}+3)'.format(length_k, n_k)


if __name__ == "__main__":
    def eval_eq(eq):
        eq += '\nprint(ans)'
        exec(eq)


    eval_eq(diff_perm(1, [22, 33], 2))
