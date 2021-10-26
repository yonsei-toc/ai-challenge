import generator.exports as gen

import functools
import itertools
import math
import random
import string

# utils
def randreal(st, ed, *, ndigits = 2):
    if ed is None:
        st, ed = 0, st

    if ndigits is None:
        return random.uniform(st, ed)
    else:
        return round(random.uniform(st, ed), ndigits=ndigits)


# it accepts an id. if it is not provided, use the function name.
# the name must be unique.
@gen.equations.register('eqn_sum')
def eqn00(*args):
    # return variable is ALWAYS [ans].
    return 'ans = sum({})'.format(repr(args))


# @gen.problems.register
def prob01(selector, tokenpool, clskey):
    # Claim items at first. They will not overlap (even for different keys).
    container = selector.get(clskey.container)
    item = selector.get(clskey.item)
    name = selector.get(clskey.name)
    count1 = random.randint(1, 100)
    count2 = random.randint(1, count1)

    # setup the words to be replaced into tokens
    container_k = tokenpool.new(container)
    item_k = tokenpool.new(item)
    # I think name is not need to be a token in this case...
    # so ignore this.
    # name_k = tokenpool.new(name)
    count1_k = tokenpool.new(count1)
    count2_k = tokenpool.new(count2)

    # str
    unit = item.of('unit')
    count1_k.unit = unit
    count2_k.unit = unit

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    # build() (is intended to) automatically substitutes tokens.
    # The token id is determined by the first variable, seperated by dots.
    # i.e., {count2.to_korunit()} uses the same token id to count2.
    # {#이} will be converted "이" or "가" according to the previous character.
    # supported:: 은, 는, 이, 가, 을, 를, 와, 과, 이?, 으?
    return gen.build(
            # body is the background of problem settings
            body=' '.join([
                '{container}{#에} {item} {count1}{#이} 있{sent_trailing}',
                '{name}{#이?}가 {container}에서 {item} {count2.to_korunit()}{#을} 꺼냈{sent_trailing}'
                # note that we did not specify a unit.
            ]),
            # question is the main sentence of the problem
            question='{container}에 있는 {item}{#은} {total}몇 {unit}{ques_trailing}',
            equation=gen.EqnRef('eqn_sum', count1_k, -count2_k),

            # specify every variables used in above strings
            env=gen.fnmap(
                container=container_k,
                item=item_k,
                name=name,
                count1=count1_k,
                count2=count2_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing,
                unit=unit
            ))


@gen.equations.register()
def eqn2(factor, result):
    return 'ans = {} // {}'.format(repr(result), repr(factor))


# @gen.problems.register
def prob02(selector, tokenpool, clskey):
    a = random.randint(2, 19)
    b = random.randint(2, 19) * a

    n1 = tokenpool.new(a)
    n2 = tokenpool.new(b)

    return gen.build(
            body='어떤 수에 {n1}{#을} 곱하면 {n2}{#가} 나온다.',
            question='어떤 수를 구하시오.',
            equation=gen.EqnRef('eqn2', n1, n2),

            env=gen.fnmap(
                n1=n1,
                n2=n2
            ))

# # @gen.problems.register
# def prob02(selector, tokenpool, clskey):
#     item = tokens.get(clskey.tool_group)
#     cvtunit = random.choice(item.of("group_unit"))
#     ans = random.randint(1, 10)

#     body = ''
#     question = f'{item} {ans * cvtunit[1]}{c(item.of("unit"), "은")} 몇 {cvtunit[0]}입니까?'

#     # do not recommend this in practice; this is for an illustration purpose..
#     variable = ''.join(random.sample('abcdefghijklmnopqrstuvwxyz', k=random.randint(1, 2)))

#     equation = f'{variable} = {ans * cvtunit[1]} // {cvtunit[1]}'

#     return template.format(body, question, equation, variable)



# You must prepend @register for each function!
# @gen.problems.register
def showcase(sel, pl, clskey):
    # get a real number from [0, 2]
    # this will round the numbers to the 1/100's digit.
    n = random.randint(2, 5)

    nums = [ randreal(0, 4, ndigits=2) for _ in range(n) ]
    nums_k = list(map(pl.new, nums))

    question = f'{gen.korutil.num2korunit(n)} 수 '
    question += ', '.join('{' + 'num{}'.format(x) + '}' for x in range(n))
    question += '의 평균은 얼마입니까?'

    envdict = { f'num{i}': nums_k[i] for i in range(n) }

    return gen.build(
            body='',
            question=question,
            equation=gen.EqnRef('avg', *nums_k),
            # if answer is not stable (due to floating point arithmetic)
            # specify a stable answer.
            answer=round(sum(nums) / n, ndigits=2),

            env=envdict)


if __name__ == '__main__':
    with open('../dict.json', 'rt') as f:
       dictionary, clskey = gen.Dictionary.load(f.read())

    for fn in gen.problems:
        i = 0
        while i < 8:
            selector = gen.DictionarySelector(dictionary)
            tokenpool = gen.TokenPool()
            ret = fn(selector, tokenpool, clskey)
            if ret is not None:
                print (ret)
                i += 1
