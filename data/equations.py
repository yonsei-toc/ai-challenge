from typing import Callable, Iterable, Optional
import random


class Equations:
    key_t = str

    def __init__(self):
        self.equation_keys = {}
        self.equations = []

    def _decorator(self, func: Callable[..., str], key: Optional[key_t] = None):
        if key is None:
            key = func.__name__

        if key in self.equation_keys:
            raise RuntimeError("Duplicated equation ID: {}.".format(key))

        self.equations.append(func)
        self.equation_keys[key] = len(self.equations) - 1
        return func

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


# it accepts an id. if it is not provided, use the function name.
# the name must be unique.
@equations.register
def max_from_n_comb(n, *L):
    """
     > {L} 중에서 서로 다른 숫자 {n}개를 뽑아 만들 수 있는 가장 큰 {n} 자리 수
    """
    return "".join([
        "L = [{}]\n".format(', '.join(map(str, L))),
        f"ans = max(filter(",
        f"lambda e: e >= 10 ** ({n} - 1),",
        f"map(",
        f"lambda e: int(''.join(map(str, e))),",
        f"itertools.permutations(L, {n}))))"])


@equations.register
def min_from_n_comb(n, *L):
    """
     > {L} 중에서 서로 다른 숫자 {n}개를 뽑아 만들 수 있는 가장 작은 {n} 자리 수
    """
    return "".join([
        "L = [{}]\n".format(', '.join(map(str, L))),
        f"ans = min(filter(",
        f"lambda e: e >= 10 ** ({n} - 1),",
        f"map(",
        f"lambda e: int(''.join(map(str, e))),",
        f"itertools.permutations(L, {n}))))"])


@equations.register
def max_diff_from_n_comb(n, *L):
    """
     > {L} 중에서 서로 다른 숫자 {n}개를 뽑아 만들 수 있는 {n} 자리 수의 차의 최대 값
    """
    return "".join([
        "L = [{}]\n".format(', '.join(map(str, L))),
        f"L = list(filter(",
        f"lambda e: e >= 10 ** ({n} - 1),",
        f"map(",
        f"lambda e: int(''.join(map(str, e))),",
        f"itertools.permutations(L, {n}))))\n",
        "ans = max(L) - min(L)"])


@equations.register
def writing_n_to_m_count_c(n, m, c):
    """
     > {n}부터 {m}까지 적었을 때 등장하는 {c}의 수
    """
    return "".join([
        "ans = len(list(",
        f"filter(lambda e: '{c}' == e, ",
        f"''.join(map(str, range({n}, {m} + 1))))",
        "))"
    ])


@equations.register
def n_comb(n, *L):
    """
     > {L} 중에서 서로 다른 숫자 {n}개를 뽑아 만들 수 있는 {n}자리 수의 수
    """
    return "".join([
        "L = [{}]\n".format(', '.join(map(str, L))),
        f"ans = math.perm(len(L) - 1, {n} - 1) * (len(L) - 1) ",
        "if 0 in L ",
        f"else math.perm(len(L), {n})"
    ])


@equations.register
def c_sum_in_range_n_is_m(c, n, m):
    """
     > 1부터 n까지의 수 중 c개의 수를 동시에 뽑아 그 합이 m이 되는 경우의 수
    """
    return "".join([
        f"L = range(1, {n})\n",
        "ans = len(list(filter(",
        f"lambda e: e == {m}, ",
        "map(",
        "sum, ",
        f"itertools.combinations(L, {c})",
        ")",
        ")))"
    ])


@equations.register
def wrong_multiplication_less(n, d, a, b, A, B):
    """
    > {n} 자리수 X, Y중 한 수의 {d}의 자리 숫자 {a}를 {b}로 잘못 보고 계산하여
    > {A}를 얻었다. 올바르게 계산한 값이 {B}일 때 X, Y 중 작은 수
    """

    # (X + ( b - a ) * d) * Y = A
    # X * Y + ( b - a ) * d * Y = A

    # X * Y = B
    # B + ( b - a ) * d * Y = A

    # Y = ( A - B ) // ( ( b - a ) * d )

    return "".join([
        f"Y = ( {A} - {B} ) // ( ( {b} - {a} ) * {d} )\n",
        f"X = {B} // Y\n",
        "ans = min(X, Y)"
    ])


@equations.register
def wrong_multiplication_greater(n, d, a, b, A, B):
    """
    > {n} 자리수 X, Y중 한 수의 {d}의 자리 숫자 {a}를 {b}로 잘못 보고 계산하여
    > {A}를 얻었다. 올바르게 계산한 값이 {B}일 때 X, Y 중 큰 수
    """

    return "".join([
        f"Y = ( {A} - {B} ) // ( ( {b} - {a} ) * {d} )\n",
        f"X = {B} // Y\n",
        "ans = max(X, Y)"
    ])


# NOTE:
# 모델이 '#1 무겁다 #2', '#1 가볍다 #2'에 따라 순서만 바꿔서 넣을 수 있도록 하겠습니다.
# 문제가 '#1 은 #2보다 가볍다. #2 는 #3보다 무겁다. 가장 가벼운 사람은 누구인가?'라면
# equation 호출은 'order_least #2 #1 #2 #3' 이런식으로.
@equations.register
def order_least(*pairs):
    """
    > pairs[2k] < pairs[2k+1]를 만족하도록 들어온다.
    """
    return "".join([
        "L = [{}]\n".format(', '.join(map(lambda e: "'" + str(e) + "'", pairs))),
        "names = { name: True for name in L }\n",
        "for i in range(0, len(L), 2): names[L[i]] = False\n",
        "ans = [ name for name in names.keys() if names[name] ][0]"
    ])


# 문제가 '#1 은 #2보다 가볍다. #2 는 #3보다 무겁다. 가장 무거운 사람은 누구입니까?'라면
# equation 호출은 'order_greatest #2 #1 #2 #3' 이런식으로.
@equations.register
def order_greatest(*pairs):
    """
    > pairs[2k] < pairs[2k+1]를 만족하도록 들어온다.
    """
    return "".join([
        "L = [{}]\n".format(', '.join(map(lambda e: "'" + str(e) + "'", pairs))),
        "names = { name: True for name in L }\n",
        "for i in range(1, len(L), 2): names[L[i]] = False\n",
        "ans = [ name for name in names.keys() if names[name] ][0]"
    ])


@equations.register
def a_div_two_sub_b(a, b):
    return f"ans = {a} // 2 - {b}"


# it accepts an id. if it is not provided, use the function name.
# the name must be unique.
@equations.register
def eqn_sum(*args):
    # return variable is ALWAYS [ans].
    return 'ans = sum([{}])'.format(', '.join(map(str, args)))


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
    return f'''cand = [(var1,var2,var3,var4) for var1 in range(10) for var2 in range(10) for var3 in range(10) for var4 in range(10) if {n1}*100 + var1*10 + {n2} + var2*100 + {n3}*10 + var3 == var4*100 + {n4}*10 + {n5} and var1!=var2!=var3!=var4]
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
def eq_c8p5(e1, e2, n1, n2):
    return f'''{e2}={n1} // (2 * ({n2}+1)); {e1} = {e2}*{n2}; ans = {e1}'''


@equations.register
def eqn_avg(*args):
    return 'ans = sum({}) / {}'.format(repr(args), len(args))


@equations.register
def max_sub_min(*args):
    # return variable is ALWAYS [ans].

    input = ','.join(list(map(str, args)))
    return 'ans = max([{}]) - min([{}])'.format(input, input)


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


@equations.register
def prob06_04(*args):
    # return variable is ALWAYS [ans].
    # args0 args1 args2
    return 'ans = round(((({}*{})+{}) / {}) - {})'.format(args[2], args[1], args[2], args[0], args[0])


@equations.register
def prob07_04(name0, name1, name2, name3, *args):
    # return variable is ALWAYS [ans].

    dict = {args[0]: name0, args[0] - args[1]: name1,
            args[2]: name2, args[0] + args[3]: name3}

    return 'ans = {}[max({}.keys())]'.format(str(dict), str(dict))


@equations.register
def prob08_04(*args):
    # return variable is ALWAYS [ans].
    # item1_k, l_k)

    if args[0] == "'정삼각형'":
        return 'ans = "%.2f" % float(({}*3) / {})'.format(args[1], 8)
    elif args[0] == "'정사각형'":
        return 'ans = "%.2f" % float(({}*4) / {})'.format(args[1], 8)
    elif args[0] == "'정오각형'":
        return 'ans = "%.2f" % float(({}*5) / {})'.format(args[1], 8)
    elif args[0] == "'정팔각형'":
        return 'ans = "%.2f" % float(({}*8) / {})'.format(args[1], 8)
    else:
        return 'ans = "%.2f" % float(({}*7) / {})'.format(args[1], 8)


@equations.register
def prob04_03(over, *args):
    # return variable is ALWAYS [ans]..
    return 'ans = list(map(lambda x: x> {}, [{}])).count(True)'.format(int(over), ','.join(list(map(str, args))))


@equations.register
def prob04_02(n, *number_list):
    """
     > {L} 중에서 서로 다른 숫자 {n}개를 뽑아 만들 수 있는 가장 작은 {n} 자리 수
    """
    return "".join([
        "L = [{}]\n".format(', '.join(map(str, number_list))),
        f"ans = min(filter(",
        f"lambda e: e >= 10 ** ({n} - 1),",
        f"map(",
        f"lambda e: int(''.join(map(str, e))),",
        f"itertools.permutations(L, {n}))))"])


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


# multiple fraction
@equations.register
def multi_frac(origin, *args):
    ans_str = 'ans = {}'.format(origin)
    for i in range(len(args)):
        if i % 2 == 0:
            ans_str += ' * {}'.format(args[i])
        else:
            ans_str += ' / {}'.format(args[i])
    return ans_str


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
