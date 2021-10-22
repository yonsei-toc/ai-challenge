import generator.exports as gen

import functools
import itertools
import math
import random
import string

"""
problems
"""
@gen.problems.register
def prob030101(sel, pl, clskey):
    """
    {nums_k} 중에서 서로 다른 숫자 {numN_k}개를 뽑아 만들 수 있는 가장 큰 {numN_k} 자리 수를 구하시오.
    """

    n = random.randint(2, 6)
    # nums = [ random.randint(0, 9) for _ in range(n) ]
    # numN = random.randint(1, n)

    # tokens
    nums_k = pl.sample(range(10), n)
    numN_k = pl.randint(2, n)

    # envdict
    envdict = {'nums': nums_k}
    # envdict = { f'num{i}': nums_k[i] for i in range(n) }
    envdict.update({ 'c': numN_k })

    body = ''

    # question = ', '.join('{' + f'num{i}' +'}' for i in range(n))
    question = '{nums}'
    question += f" 중{random.choice(['','에서'])} "
    question += random.choice([
        "서로 다른 숫자 {c.to_korunit()} 개",
        "{c.to_korunit()} 개의 서로 다른 숫자",
        "서로 다른 {c.to_korunit()} 개의 숫자",
        "서로 다른 {c.to_korunit()} 숫자",

        "서로 다른 숫자 {c}개",
        "{c}개의 서로 다른 숫자",
        "서로 다른 {c}개의 숫자"
    ])
    question += "를 뽑아 "
    question += random.choice([
        "만들 수 있는 {c.to_korunit()} 자리 수 중 가장 큰 수",
        "만들 수 있는 가장 큰 {c.to_korunit()} 자리 수"
    ])
    question += random.choice([
        "를 구하시오.", 
        "는 얼마입니까?", 
        "를 쓰시오."
    ])

    equation = gen.EqnRef("diff_perm", numN_k, nums_k, 0)

    return gen.build(
            body='', 
            question=question, 
            equation=equation, 

            env=envdict)

@gen.problems.register
def prob030102(sel, pl, clskey):
    """
    {nums_k} 중에서 서로 다른 숫자 {numN_k}개를 뽑아 만들 수 있는 가장 작은 {numN_k} 자리 수를 구하시오.
    """

    n = random.randint(2, 6)
    # nums = [ random.randint(0, 9) for _ in range(n) ]
    # numN = random.randint(1, n)

    # tokens
    nums_k = pl.sample(range(10), n)
    numN_k = pl.randint(2, n)

    # envdict
    # envdict = { f'num{i}': nums_k[i] for i in range(n) }
    envdict = {'nums': nums_k}
    envdict.update({ 'c': numN_k })

    body = ''

    # question = ', '.join('{' + f'num{i}' +'}' for i in range(n))
    question = '{nums}'
    question += f" 중{random.choice(['','에서'])} "
    question += random.choice([
        "서로 다른 숫자 {c.to_korunit()} 개",
        "{c.to_korunit()} 개의 서로 다른 숫자",
        "서로 다른 {c.to_korunit()} 개의 숫자",
        "서로 다른 {c.to_korunit()} 숫자",

        "서로 다른 숫자 {c}개",
        "{c}개의 서로 다른 숫자",
        "서로 다른 {c}개의 숫자"
    ])
    question += "를 뽑아 "
    question += random.choice([
        "만들 수 있는 {c.to_korunit()} 자리 수 중 가장 작은 수",
        "만들 수 있는 가장 작은 {c.to_korunit()} 자리 수"
    ])
    question += random.choice([
        "를 구하시오.", 
        "는 얼마입니까?", 
        "를 쓰시오."
    ])

    equation = gen.EqnRef("diff_perm", numN_k, nums_k, 1)

    return gen.build(
            body='', 
            question=question, 
            equation=equation, 

            env=envdict)

@gen.problems.register
def prob030103(sel, pl, clskey):
    """
    바구니에 1, 2, 3 이 적혀있는 카드가 들어있다.
    바구니에서 카드를 꺼내서 만들 수 있는 가장 큰 두 자리 수를 구하시오.
    """

    n = random.randint(2, 7)

    # items
    container = sel.get(clskey.container)
    item = sel.get(clskey.writable)

    # tokens
    nums_k = pl.sample(range(10), n)
    numN_k = pl.randint(2, n)

    # unit
    # numN_k.unit = item.of('unit')

    # trailing
    body_trailing, question_trailing = random.choice([
        ("있다.", "를 구하시오."),
        ("있다.", "를 쓰시오."),
        ("있다.", "는 얼마인가?"),

        ("있습니다.", "는 얼마입니까?")
    ])

    # envdict
    # envdict = { f'num{i}': nums_k[i] for i in range(n) }
    envdict = {'nums': nums_k}
    envdict.update({ 'c': numN_k })
    envdict.update({ 'container': container })
    envdict.update({ 'item': item })

    # body
    body = '{container}에 '
    body += random.choice(["숫자 ", ""])
    # body += ', '.join('{' + f'num{i}' +'}' for i in range(n))
    body += '{nums}'
    body += '{#이} '
    body += random.choice([
        "적혀 있는 ", # 둘 다 표준임...
        "적혀있는 ",
        "적힌 ",
        "쓰여 있는 ",
        "쓰여있는 ",
        "쓰인 "
    ])
    body += "{item}{#가} "
    body += random.choice([
        "들어",
        ""
    ])
    body += body_trailing

    # question
    question = random.choice([
        "{container}{#에서} "
        ""
    ])
    question += random.choice([
        "{item}{#을} "
        ""
    ])
    question += random.choice([
        "뽑아 ",
        "꺼내서 ",
        "꺼내 "
    ])
    question += random.choice([
        "만들 수 있는 {c.to_korunit()} 자리 수 중 가장 큰 수",
        "만들 수 있는 가장 큰 {c.to_korunit()} 자리 수"
    ])
    question += question_trailing

    # equation
    equation = gen.EqnRef("diff_perm", numN_k, nums_k, 0)

    return gen.build(
            body=body, 
            question=question, 
            equation=equation, 

            env=envdict)

@gen.problems.register
def prob030104(sel, pl, clskey):
    """
    바구니에 1, 2, 3 이 적혀있는 카드가 들어있다.
    바구니에서 카드를 꺼내서 만들 수 있는 가장 작은 두 자리 수를 구하시오.
    """

    n = random.randint(2, 7)

    # items
    container = sel.get(clskey.container)
    item = sel.get(clskey.writable)

    # tokens
    nums_k = pl.sample(range(10), n)
    numN_k = pl.randint(2, n)

    # unit
    # numN_k.unit = item.of('unit')

    # trailing
    body_trailing, question_trailing = random.choice([
        ("있다.", "를 구하시오."),
        ("있다.", "를 쓰시오."),
        ("있다.", "는 얼마인가?"),

        ("있습니다.", "는 얼마입니까?")
    ])

    # envdict
    # envdict = { f'num{i}': nums_k[i] for i in range(n) }
    envdict = {'nums': nums_k}
    envdict.update({ 'c': numN_k })
    envdict.update({ 'container': container })
    envdict.update({ 'item': item })

    # body
    body = '{container}에 '
    body += random.choice(["숫자 ", ""])
    # body += ', '.join('{' + f'num{i}' +'}' for i in range(n))
    body += '{nums}'
    body += '{#이} '
    body += random.choice([
        "적혀 있는 ",
        "적혀있는 ",
        "적힌 ",
        "쓰여 있는 ",
        "쓰여있는 ",
        "쓰인 "
    ])
    body += "{item}{#가} "
    body += random.choice([
        "들어",
        ""
    ])
    body += body_trailing

    # question
    question = random.choice([
        "{container}{#에서} "
        ""
    ])
    question += random.choice([
        "{item}{#을} "
        ""
    ])
    question += random.choice([
        "뽑아 ",
        "꺼내서 ",
        "꺼내 "
    ])
    question += random.choice([
        "만들 수 있는 {c.to_korunit()} 자리 수 중 가장 작은 수",
        "만들 수 있는 가장 작은 {c.to_korunit()} 자리 수"
    ])
    question += question_trailing

    # equation
    equation = gen.EqnRef("diff_perm", numN_k, nums_k, 1)

    return gen.build(
            body=body, 
            question=question, 
            equation=equation, 

            env=envdict)

# TODO
# @gen.problems.register
def prob030105(sel, pl, clskey):
    """
    바구니에 1, 2, 3 이 적혀있는 카드가 들어있다.
    바구니에서 카드를 두 장 꺼내서 만들 수 있는 가장 큰 (두 자리) 수를 구하시오.
    """
    return

@gen.problems.register
def prob030201(sel, pl, clskey):
    """
    {nums_k} 중에서 서로 다른 숫자 {numN_k}개를 뽑아 만들 수 있는 
    {numN_k} 자리 수 중에서 가장 큰 수와 가장 작은 수의 차를 구하시오.
    """

    n = random.randint(2, 6)
    # nums = [ random.randint(0, 9) for _ in range(n) ]
    # numN = random.randint(1, n)

    # tokens
    nums_k = pl.sample(range(10), n)
    numN_k = pl.randint(2, n)

    # envdict
    # envdict = { f'num{i}': nums_k[i] for i in range(n) }
    envdict = {'nums': nums_k}
    envdict.update({ 'c': numN_k })

    body = ''

    # question = ', '.join('{' + f'num{i}' +'}' for i in range(n))
    question = '{nums}'
    question += f" 중{random.choice(['','에서'])} "
    question += random.choice([
        "서로 다른 숫자 {c.to_korunit()} 개",
        "{c.to_korunit()} 개의 서로 다른 숫자",
        "서로 다른 {c.to_korunit()} 개의 숫자",
        "서로 다른 {c.to_korunit()} 숫자",

        "서로 다른 숫자 {c}개",
        "{c}개의 서로 다른 숫자",
        "서로 다른 {c}개의 숫자"
    ])
    question += "를 뽑아 만들 수 있는 "
    question += random.choice([
        "{c.to_korunit()} 자리 수 중 ",
        "{c.to_korunit()} 자리 수 중에서 ",
        "수 중에서 ",
        "수 중 ",
        ""
    ])
    question += random.choice([
        "가장 큰 수와 가장 작은 수의 차",
        "가장 큰 수와 가장 작은 수의 차이",
        "가장 작은 수와 가장 큰 수의 차",
        "가장 작은 수와 가장 큰 수의 차이",
        "가장 큰 수에서 가장 작은 수를 뺀 값"
    ])
    question += random.choice([
        "{#를} 구하시오.", 
        "{#는} 얼마입니까?", 
        "{#를} 쓰시오."
    ])

    equation = gen.EqnRef("diff_perm", numN_k, nums_k, 2)

    return gen.build(
            body='', 
            question=question, 
            equation=equation, 

            env=envdict)

@gen.problems.register
def prob030202(sel, pl, clskey):
    """
    바구니에 1, 2, 3 이 적혀있는 카드가 들어있다.
    바구니에서 카드를 몇 장 꺼내서 만들 수 있는 가장 큰 수와 작은 수의 차를 구하시오.
    """

    n = random.randint(2, 7)

    # items
    container = sel.get(clskey.container)
    item = sel.get(clskey.writable)

    # tokens
    nums_k = pl.sample(range(10), n)
    numN_k = pl.randint(2, n)

    # unit
    # numN_k.unit = item.of('unit')

    # envdict
    # envdict = { f'num{i}': nums_k[i] for i in range(n) }
    envdict = {'nums': nums_k}
    envdict.update({ 'c': numN_k })
    envdict.update({ 'container': container })
    envdict.update({ 'item': item })

    # trailings
    body_trailing, question_trailing = random.choice([
        ("습니다.", "입니까?"),
        ("다.", "인가?")
    ])

    # body
    body = '{container}에 '
    body += random.choice(["숫자 ", ""])
    # body += ', '.join('{' + f'num{i}' +'}' for i in range(n))
    body += '{nums}'
    body += '{#이} '
    body += random.choice([
        "적혀 있는 ",
        "적혀있는 ",
        "적힌 ",
        "쓰여 있는 ",
        "쓰여있는 ",
        "쓰인 "
    ])
    body += "{item}{#가} 들어있" + body_trailing

    # question
    question = random.choice([
        "{container}{#에서} "
        ""
    ])
    question += random.choice([
        "{item}{#을} "
        ""
    ])
    question += random.choice([
        "뽑아 ",
        "꺼내서 ",
        "꺼내 "
    ])
    question += "만들 수 있는 {c} 자리 수 중에서 "
    question += random.choice([
        "가장 큰 수와 가장 작은 수의 차",
        "가장 큰 수와 가장 작은 수의 차이",
        "가장 작은 수와 가장 큰 수의 차",
        "가장 작은 수와 가장 큰 수의 차이",
        "가장 큰 수에서 가장 작은 수를 뺀 값"
    ])
    question += random.choice([
        "{#를} 구하시오.", 
        "{#를} 쓰시오.",

        "{#는} 얼마"+question_trailing,
    ])

    # equation
    equation = gen.EqnRef("diff_perm", numN_k, nums_k, 2)

    return gen.build(
            body=body, 
            question=question, 
            equation=equation, 

            env=envdict)
# TODO : diff_perm 에서 {1} 대신에 range 가 들어가는 문제 만들기
@gen.problems.register
def prob030301(sel, pl, clskey):
    """
    1부터 30까지 자연수를 쓰려고 합니다. 숫자 2는 모두 몇 번 써야 합니까?
    """

    m = random.randint(2, 100)
    n = random.randint(1, m)
    c = random.randint(0, 9)

    # tokens
    m_k = pl.new(m)
    n_k = pl.new(n)
    c_k = pl.new(c)

    # envdict
    envdict = {}
    envdict.update({ 'm' : m_k })
    envdict.update({ 'n' : n_k })
    envdict.update({ 'c' : c_k })

    # trailing
    body_trailing, question_trailing = random.choice([
        ("합니다.", "합니까?"),
        ("한다.", "하는가?")
    ])

    # body
    body = random.choice([
        "{n}부터 {m}까지 ",
        "{n} 이상 {m} 이하의 "
    ])
    body += random.choice([
        "숫자를 ",
        "자연수를 ",
    ])
    body += random.choice([
        "모두 ",
        "",
    ])
    body += random.choice([
        "적으려고 ",
        "쓰려고 "
    ])
    body += body_trailing

    # question
    question = random.choice([
        "숫자 ",
        ""
    ])
    question += "{c}{#는}"
    question += random.choice([
        "모두 ", 
        " "
    ])
    question += random.choice([
        "몇 번 써야 ",
        "몇 번 적어야 ",
        "몇 번 등장"
    ])
    question += question_trailing

    # equation
    equation = gen.EqnRef("count_from_range", n_k, m_k, c_k)

    return gen.build(
            body=body, 
            question=question, 
            equation=equation, 

            env=envdict)

@gen.problems.register
def prob030401(sel, pl, clskey):
    """
    0, 2, 4, 6, 8 중 서로 다른 2개의 숫자를 뽑아서 두 자리 수를 만들려고 합니다.
    만들 수 있는 두 자리 수는 모두 몇 개입니까?
    """

    L = random.randint(2, 6)
    # nums = [ random.randint(0, 9) for _ in range(n) ]
    # numN = random.randint(1, n)

    # tokens
    ns_k = pl.sample(range(10), L)
    c_k = pl.randint(1, L)

    # envdict
    # envdict = { f'n{i}': ns_k[i] for i in range(L) }
    envdict = {'ns': ns_k}
    envdict.update({ 'c': c_k })

    # trailing
    body_trailing, question_trailing = random.choice([
        ("합니다.", "입니까?"),
        ("한다.", "인가?"),
        ("한다.", "인지 구하시오.")
    ])

    body = ''

    # body += ', '.join('{' + f'n{i}' +'}' for i in range(L))
    body += '{ns}'
    body += f" 중{random.choice(['','에서'])} "
    body += random.choice([
        "서로 다른 숫자 {c.to_korunit()} 개",
        "{c.to_korunit()} 개의 서로 다른 숫자",
        "서로 다른 {c.to_korunit()} 개의 숫자",
        "서로 다른 {c.to_korunit()} 숫자",

        "서로 다른 숫자 {c}개",
        "{c}개의 서로 다른 숫자",
        "서로 다른 {c}개의 숫자"
    ])
    body += "를 뽑아" 
    body += random.choice([
        "서 ",
        " "
    ])
    body += "{c.to_korunit()} 자리 수를 " 
    body += random.choice([
        "만드려 ",
        "만들려고 "
    ])
    body += body_trailing

    question = random.choice([
        "만들 수 있는 ",
        "가능한 "
    ])
    question += random.choice([
        "{c.to_korunit()} 자리 수는 ",
        "수는 "
    ])
    question += random.choice([
        "모두 ",
        ""
    ])
    question += "몇 개"
    question += question_trailing

    equation = gen.EqnRef("diff_perm", c_k, ns_k, 3)

    return gen.build(
            body=body, 
            question=question, 
            equation=equation, 

            env=envdict)
# TODO
# @gen.problems.register
def prob030401(sel, pl, clskey):
    """
    사과, 복숭아, 배, 참외 중 2가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?
    """
    return

@gen.problems.register
def prob030501(sel, pl, clskey):
    """
    10보다 작은 자연수 중에서 서로 다른 세 수를 동시에 뽑으려고 합니다.
    세 수의 합이 12인 경우의 수를 구하시오.
    """

    n = random.randint(2, 20)
    c = random.randint(2, min(5, n))

    # tokens
    n_k = pl.new(n)
    c_k = pl.new(c)
    m_k = pl.randint((c * (c+1)) // 2, c * n - (c * (n-1)) // 2)

    # envdict
    envdict = {}
    envdict.update({ 'n' : n_k })
    envdict.update({ 'c' : c_k })
    envdict.update({ 'm' : m_k })

    # trailing
    body_trailing, question_trailing = random.choice([
        ("합니다.", "{#를} 구하시오."),
        ("합니다.", "{#를} 쓰시오."),
        ("합니다.", "{#는} 얼마입니까?"),

        ("한다.", "{#를} 구하시오."),
        ("한다.", "{#를} 쓰시오."),
        ("한다.", "{#는} 얼마인가?")
    ])

    # body
    body = "{n}보다 작은 "
    body += random.choice([
        "양의 정수 ",
        "자연수 "
    ])
    body += "중에서 서로 다른 "
    body += "{c.to_korunit()} "
    body += random.choice([
        "개의 ",
        ""
    ])
    body += "수를 "
    body += random.choice([
        "동시에 ",
        ""
    ])
    body += random.choice([
        "뽑으려고 " + body_trailing,
        "고르려고 " + body_trailing,
        "뽑으려 " + body_trailing,
        "고르려 " + body_trailing
    ])

    # question
    question = "{c.to_korunit()} "
    question += random.choice([
        "개의 ",
        ""
    ])
    question += "수의 합이 {m}인 경우의 수"
    question += question_trailing

    # equation
    equation = gen.EqnRef("find_sum_from_range", 0, c_k, n_k, m_k)

    return gen.build(
            body=body, 
            question=question, 
            equation=equation, 

            env=envdict)

# TODO
# @gen.problems.register
def prob030502(sel, pl, clskey):
    """
    10, 20, 32, 4 중에서 서로 다른 세 수를 동시에 뽑으려고 합니다.
    세 수의 합이 12인 경우의 수를 구하시오.
    """
    return

# TODO : X, Y 문제
@gen.problems.register
def prob060301(sel, pl, clskey):
    """
    두 자리 수끼리의 곱셈에서 곱하는 수의 십의 자리 숫자
    2를 6으로 잘못 보고 계산한 값이 2432가 되었습니다.
    바르게 계산한 값이 912일 때, 2개의 두 자리 수 중 더 작은 수를 쓰시오.
    """

    n = random.randint(2, 3) # n 자리 수

    X = random.randint(10 ** (n - 1), 10 ** n - 1)
    Y = random.randint(10 ** (n - 1), 10 ** n - 1)

    d = 10 ** random.randint(0, n - 1) # d의 자리 숫자

    a = X // d % 10
    b = random.choice([ e for e in range(10) if e != a ])

    A = ( X + (b - a) * d ) * Y
    B = X * Y

    # tokens
    n_k = pl.new(n)
    d_k = pl.new(d)
    a_k = pl.new(a)
    b_k = pl.new(b)
    A_k = pl.new(A)
    B_k = pl.new(B)

    # envdict
    envdict = {}
    envdict.update({ 'n' : n_k })
    envdict.update({ 'd' : d_k })
    envdict.update({ 'a' : a_k })
    envdict.update({ 'b' : b_k })
    envdict.update({ 'A' : A_k })
    envdict.update({ 'B' : B_k })

    body_trailing, question_trailing = random.choice([
        (["입니다.", "습니다."], "{#를} 구하시오."),
        (["입니다.", "습니다."], "{#를} 쓰시오."),
        (["입니다.", "습니다."], "{#는} 얼마입니까?"),

        (["이다.", "다."], "{#를} 구하시오."),
        (["이다.", "다."], "{#를} 쓰시오."),
        (["이다.", "다."], "{#는} 얼마인가?"),
    ])

    # body
    body = "{n.to_korunit()} 자리 수끼리의 곱셈에서 "
    body += random.choice([
        "곱하는 ",
        "곱해지는 ",
        "한 "
    ])
    body += "수의 {d.to_kor()}의 자리 숫자 {a}{#을} {b}{#으?}로 잘못 보고 "
    body += random.choice([
        random.choice([
            f"계산한 값{random.choice(['은','이'])} ",
            f"계산한 결과{random.choice(['는','가'])} "
        ]) + random.choice([
            "{A}"+body_trailing[0],
            "{A}{#가} 되었"+body_trailing[1]
        ]),
        "계산하여 {A}{#를} 얻었"+body_trailing[1]
    ])

    # question
    question = "바르게 계산한 "
    question += random.choice([
        random.choice([
            f"계산한 값{random.choice(['은','이'])} ",
            f"계산한 결과{random.choice(['는','가'])} "
        ]) + random.choice([
            "{B}일 때, ",
            "{B}{#가} 될 때, "
        ]),
        "계산하여 {B}{#를} 얻을 때, "
    ])
    question += "두 개의 {n.to_korunit()} 자리 수 중 "
    question += random.choice([
        "더 ", 
        ""
    ])
    question += "작은 수"
    question += question_trailing

    # equation
    equation = gen.EqnRef("wrong_multiply", n_k, d_k, a_k, b_k, A_k, B_k, 0)

    return gen.build(
            body=body, 
            question=question, 
            equation=equation, 

            env=envdict)

@gen.problems.register
def prob060302(sel, pl, clskey):
    """
    두 자리 수끼리의 곱셈에서 곱하는 수의 십의 자리 숫자
    2를 6으로 잘못 보고 계산한 값이 2432가 되었습니다.
    바르게 계산한 값이 912일 때, 2개의 두 자리 수 중 더 큰 수를 쓰시오.
    """

    n = random.randint(2, 3) # n 자리 수

    X = random.randint(10 ** (n - 1), 10 ** n - 1)
    Y = random.randint(10 ** (n - 1), 10 ** n - 1)

    d = 10 ** random.randint(0, n - 1) # d의 자리 숫자

    a = X // d % 10
    b = random.choice([ e for e in range(10) if e != a ])

    A = ( X + (b - a) * d ) * Y
    B = X * Y

    # tokens
    n_k = pl.new(n)
    d_k = pl.new(d)
    a_k = pl.new(a)
    b_k = pl.new(b)
    A_k = pl.new(A)
    B_k = pl.new(B)

    # envdict
    envdict = {}
    envdict.update({ 'n' : n_k })
    envdict.update({ 'd' : d_k })
    envdict.update({ 'a' : a_k })
    envdict.update({ 'b' : b_k })
    envdict.update({ 'A' : A_k })
    envdict.update({ 'B' : B_k })

    body_trailing, question_trailing = random.choice([
        (["입니다.", "습니다."], "{#를} 구하시오."),
        (["입니다.", "습니다."], "{#를} 쓰시오."),
        (["입니다.", "습니다."], "{#는} 얼마입니까?"),

        (["이다.", "다."], "{#를} 구하시오."),
        (["이다.", "다."], "{#를} 쓰시오."),
        (["이다.", "다."], "{#는} 얼마인가?"),
    ])

    # body
    body = "{n.to_korunit()} 자리 수끼리의 곱셈에서 "
    body += random.choice([
        "곱하는 ",
        "곱해지는 ",
        "한 "
    ])
    body += "수의 {d.to_kor()}의 자리 숫자 {a}{#을} {b}{#으?}로 잘못 보고 "
    body += random.choice([
        random.choice([
            f"계산한 값{random.choice(['은','이'])} ",
            f"계산한 결과{random.choice(['는','가'])} "
        ]) + random.choice([
            "{A}"+body_trailing[0],
            "{A}{#가} 되었"+body_trailing[1]
        ]),
        "계산하여 {A}{#를} 얻었"+body_trailing[1]
    ])

    # question
    question = "바르게 계산한 "
    question += random.choice([
        random.choice([
            f"계산한 값{random.choice(['은','이'])} ",
            f"계산한 결과{random.choice(['는','가'])} "
        ]) + random.choice([
            "{B}일 때, ",
            "{B}{#가} 될 때, "
        ]),
        "계산하여 {B}{#를} 얻을 때, "
    ])
    question += "두 개의 {n.to_korunit()} 자리 수 중 "
    question += random.choice([
        "더 ",
        ""
    ])
    question += "큰 수"
    question += question_trailing

    # equation
    equation = gen.EqnRef("wrong_multiply", n_k, d_k, a_k, b_k, A_k, B_k, 1)

    return gen.build(
            body=body,
            question=question,
            equation=equation,

            env=envdict)

# NOTE: This can generate a problem with a non-unique answer
# NOTE: #1, #2가 body에서 언급되는 순서와 무관하게 무작위로 매겨짐.
# @gen.problems.register
def prob070301(sel, pl, clskey):
    """
    석진이는 호석이보다 무겁고 지민이보다 가볍습니다. 
    남준이는 지민이보다 무겁습니다. 
    4명 중 가장 가벼운 사람은 누구입니까?
    """

    n = random.randint(3, 5)

    # items
    order = sel.get(clskey.ord_rel)

    # tokens
    n_k = pl.new(n)
    names_k = pl.sample([ sel.get(clskey.name) for _ in range(n) ], n)

    # envdict
    # envdict = { f'name{i}': names_k[i] for i in range(n) }
    envdict = {'names': names_k}
    envdict.update({ 'n': n_k })
    envdict.update({ 'order': order })

    pairs = list(itertools.combinations(range(n), 2))
    random.shuffle(pairs)
    pairs = pairs[:n * n // 4]

    body = ""

    # order variation...
    order, order_of_reverse, question_trailing = random.choice([
        ("{order}", "{order.of('reverse')}", "{#은} 누구인가?"),
        ("{order.of('var')}", "{order.of('reverse_var')}", "{#은} 누구입니까?")
    ])

    flag = False
    for n, (i, j) in enumerate(pairs):
        if n == len(pairs)-1 or flag:
            line = random.choice([
                f"{{name{i}}}{{#는}} {{name{j}}}보다 {order}다. ",
                f"{{name{j}}}{{#는}} {{name{i}}}보다 {order_of_reverse}다. "
            ])
            flag = False
        else:
            if random.choice([0, 1]) == 0:
                line = random.choice([
                    f"{{name{i}}}{{#는}} {{name{j}}}보다 {order}고 ",
                    f"{{name{j}}}{{#는}} {{name{i}}}보다 {order_of_reverse}고 "
                ])
                flag = True
            else:
                line = random.choice([
                    f"{{name{i}}}{{#는}} {{name{j}}}보다 {order}다. ",
                    f"{{name{j}}}{{#는}} {{name{i}}}보다 {order_of_reverse}다. "
                ])
                flag = False
        body += line

    question = random.choice(["{n}명 중 ", f""])
    question += "가장 {order.of('reverse_adv')} 사람"
    question += random.choice([
        "을 구하시오.",
        question_trailing
    ])

    equation = gen.EqnRef(
        "order_by_comp", 0,
        *[ names_k[pair[i]] for pair in pairs for i in range(2) ])

    return gen.build(
            body=body, 
            question=question, 
            equation=equation, 

            env=envdict)

# @gen.problems.register
def prob070302(sel, pl, clskey):
    """
    석진이는 호석이보다 무겁고 지민이보다 가볍습니다. 
    남준이는 지민이보다 무겁습니다. 
    4명 중 가장 무거운 사람은 누구입니까?
    """

    n = random.randint(3, 5)

    # items
    order = sel.get(clskey.ord_rel)

    # tokens
    n_k = pl.new(n)
    names_k = pl.sample([ sel.get(clskey.name) for _ in range(n) ], n)

    # envdict
    envdict = { f'name{i}': names_k[i] for i in range(n) }
    envdict.update({ 'n': n_k })
    envdict.update({ 'order': order })

    pairs = list(itertools.combinations(range(n), 2))
    random.shuffle(pairs)
    pairs = pairs[:n * n // 4]

    body = ""

    # order variation...
    order, order_of_reverse, question_trailing = random.choice([
        ("{order}", "{order.of('reverse')}", "{#은} 누구인가?"),
        ("{order.of('var')}", "{order.of('reverse_var')}", "{#은} 누구입니까?")
    ])

    flag = False
    for n, (i, j) in enumerate(pairs):
        if n == len(pairs)-1 or flag:
            line = random.choice([
                f"{{name{i}}}{{#는}} {{name{j}}}보다 {order}다. ",
                f"{{name{j}}}{{#는}} {{name{i}}}보다 {order_of_reverse}다. "
            ])
            flag = False
        else:
            if random.choice([0, 1]) == 0:
                line = random.choice([
                    f"{{name{i}}}{{#는}} {{name{j}}}보다 {order}고 ",
                    f"{{name{j}}}{{#는}} {{name{i}}}보다 {order_of_reverse}고 "
                ])
                flag = True
            else:
                line = random.choice([
                    f"{{name{i}}}{{#는}} {{name{j}}}보다 {order}다. ",
                    f"{{name{j}}}{{#는}} {{name{i}}}보다 {order_of_reverse}다. "
                ])
                flag = False
        body += line

    question = random.choice(["{n}명 중 ", f""])
    question += "가장 {order.of('adv')} 사람"
    question += random.choice([
        "을 구하시오.",
        question_trailing
    ])

    equation = gen.EqnRef(
        "order_by_comp", 1,
        *[ names_k[pair[i]] for pair in pairs for i in range(2) ])

    return gen.build(
            body=body, 
            question=question, 
            equation=equation, 

            env=envdict)

# @gen.problems.register
def prob080301(sel, pl, clskey):
    """
    길이가 20cm인 철사로 직사각형을 만들었더니 철사가 남지도 모자라지도 않았습니다.
    직사각형의 가로 길이가 4cm일 때, 세로 길이는 몇 cm입니까?
    """
    a = random.randint(1, 1000) * 2
    b = random.randint(1, a // 2)

    # items
    unit = sel.get(clskey.length_unit)
    unit_str = random.choice([
        unit.text,
        unit.of('symbol'),
        unit.of('kor')
    ])
    # unit = sel.get(clskey.length_unit).of
    string = sel.get(clskey.wire)

    # tokens
    a_k = pl.new(a)
    a_k.unit = unit_str

    b_k = pl.new(b)
    b_k.unit = unit_str

    # envdict
    envdict = { 'a': a_k, 'b': b_k, 'unit': unit, 'string': string }

    body_trailing, question_trailing = random.choice([
        ("다.", "인가?"),
        ("습니다.", "입니까?")
    ])

    body = random.choice([
        "길이가 {a}인 {string}{#으?}로 직사각형을 만들었더니, {string}{#가} " + random.choice([
            "남지도 모자라지도 않았"+body_trailing,
            "모자라지도 남지도 않았"+body_trailing,
            "딱 맞아 떨어졌"+body_trailing,
        ]),
        "길이가 {a}인 {string}{#을} 모두 사용하여 직사각형을 만들었" + body_trailing
    ])

    question = random.choice([
        "직사각형의 가로 길이가 {b}일 때, 세로 길이는 몇 {unit}",
        "직사각형의 세로 길이가 {b}일 때, 가로 길이는 몇 {unit}"
    ])
    question += random.choice([
        "인지 구하시오.",
        question_trailing
    ])

    # equation
    equation = gen.EqnRef("half_sub", a_k, b_k)

    return gen.build(
            body=body, 
            question=question, 
            equation=equation, 

            env=envdict)

# TODO
# @gen.problems.register
def prob080302(sel, pl, clskey):
    """
    길이가 15.7cm인 철사로 직사각형을 만들었더니 철사가 남지도 모자라지도 않았습니다.
    직사각형의 가로 길이가 3.2cm일 때, 세로 길이는 몇 cm입니까?
    """
    return

if __name__ == '__main__':
    with open('dict.json', 'rt') as f:
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
