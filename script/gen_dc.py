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

######################################################################################
################################## problem 01 ########################################


# @gen.problems.register
def prob02_1_1(selector, tokenpool, clskey):
    ''' 학교에서 국어, 수학, 영어, 과학, 사회의 순서로 시험을 봤습니다.
    3번째로 시험을 본 과목은 무엇입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent2 = random.choice([' ', '시험을 '])
    sent3 = random.choice(['순서로 ', '순으로 ', '순서대로 '])

    # new variables
    n=random.randint(2,5)
    place = selector.get(clskey.school_group)
    subjects=[selector.get(clskey.subject) for _ in range(n)]
    name_idx = random.randint(1, len(subjects))
    num1 = random.choice([name_idx, gen.korutil.num2korord(name_idx)])

    num1_k = tokenpool.new(num1)
    name_idx_k = tokenpool.new(name_idx)
    subjects_k = list(map(tokenpool.new, subjects))

    body = '{place}에서 '
    body += f', '.join('{' + 'subjects{}'.format(x) + '}' for x in range(n))
    body += '의 {sent3}시험을 봤{sent_trailing}'
    subject_dict = { f'subjects{i}': subjects_k[i] for i in range(n) }
    envdict = gen.fnmap(
                num1=num1_k,
                name_idx=name_idx_k,
                place=place,
                subjects=subjects_k,
                sent2=sent2,
                sent3=sent3,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            )
    envdict.update(subject_dict)

    return gen.build(
            body=body,
            question=f'{random.choice([name_idx, gen.korutil.num2korord(name_idx)])}번째로 '\
            '{sent2}본 과목은 무엇{ques_trailing}',
            equation=gen.EqnRef('eqn02_1', subjects_k, n_k),
            env=envdict
            )


# @gen.problems.register
def prob02_1_2(selector, tokenpool, clskey):
    ''' 학교에서 국어, 수학, 영어, 과학, 사회의 순서로 시험을 봤습니다. 
    어떤 과목을 3번째로 보았습니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['습니까?', '는지 구하시오.', '는가?'])
    sent1 = random.choice(['어떤 ', '무슨 '])
    sent3 = random.choice(['순서로 ', '순으로 ', '순서대로 '])

    # new variables
    n=random.randint(2,5)
    place = selector.get(clskey.school_group)
    subjects=[selector.get(clskey.subject) for _ in range(n)]
    name_idx = random.randint(1, len(subjects))
    num1 = random.choice([name_idx, gen.korutil.num2korord(name_idx)])

    num1_k = tokenpool.new(num1)
    name_idx_k = tokenpool.new(name_idx)
    subjects_k = list(map(tokenpool.new, subjects))

    body = '{place}에서 '
    body += f', '.join('{' + 'subjects{}'.format(x) + '}' for x in range(n))
    body += '의 {sent3}시험을 봤{sent_trailing}'
    subject_dict = { f'subjects{i}': subjects_k[i] for i in range(n) }
    envdict = gen.fnmap(
                num1=num1_k,
                name_idx=name_idx_k,
                place=place,
                subjects=subjects_k,
                sent1=sent1,
                sent3=sent3,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            )
    envdict.update(subject_dict)

    return gen.build(
            body=body,
            question='{sent1} 과목을 '\
            f'{random.choice([name_idx, gen.korutil.num2korord(name_idx)])}번째로 '\
            '보았{ques_trailing}',
            equation=gen.EqnRef('eqn02_1', subjects_k, n_k),
            env=envdict
            )

# @gen.problems.register
def prob02_1_3(selector, tokenpool, clskey):
    ''' 정국이는 국어, 수학, 영어, 과학, 사회의 순서로 시험을 봤습니다. 
    정국이가 3번째로 시험을 본 과목은 무엇입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent1 = random.choice(['어떤 ', '무슨 '])
    sent2 = random.choice(['', '시험을 '])
    sent3 = random.choice(['순서로 ', '순으로 ', '순서대로 '])
    name = selector.get(clskey.name)

    # new variables
    n=random.randint(2,5)
    place = selector.get(clskey.school_group)
    subjects=[selector.get(clskey.subject) for _ in range(n)]
    name_idx = random.randint(1, len(subjects))
    num1 = random.choice([name_idx, gen.korutil.num2korord(name_idx)])

    num1_k = tokenpool.new(num1)
    name_idx_k = tokenpool.new(name_idx)
    subjects_k = list(map(tokenpool.new, subjects))

    body = '{name}{#이?}는 '
    body += f', '.join('{' + 'subjects{}'.format(x) + '}' for x in range(n))
    body += '의 {sent3}시험을 봤{sent_trailing}'
    subject_dict = { f'subjects{i}': subjects_k[i] for i in range(n) }
    envdict = gen.fnmap(
                num1=num1_k,
                name_idx=name_idx_k,
                place=place,
                subjects=subjects_k,
                name=name,
                sent1=sent1,
                sent2=sent2,
                sent3=sent3,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            )
    envdict.update(subject_dict)

    return gen.build(
            body=body,
            question='{name}{#이?}가 '\
            f'{random.choice([name_idx, gen.korutil.num2korord(name_idx)])}번째로 '\
            '{sent2}본 과목은 무엇{ques_trailing}',
            equation=gen.EqnRef('eqn02_1', subjects_k, name_idx_k),
            env=envdict
            )

# @gen.problems.register
def prob02_1_4(selector, tokenpool, clskey):
    ''' 학교에서 국어, 수학, 영어, 과학, 사회의 순서로 시험을 봤을 때,
    3번째로 본 과목은 무슨 과목인지 구하시오 '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent1 = random.choice(['어떤 ', '무슨 '])
    sent2 = random.choice(['', '시험을 '])
    sent3 = random.choice(['순서로 ', '순으로 ', '순서대로 '])
    name = selector.get(clskey.name)

    # new variables
    n=random.randint(2,5)
    place = selector.get(clskey.school_group)
    subjects=[selector.get(clskey.subject) for _ in range(n)]
    name_idx = random.randint(1, len(subjects))
    num1 = random.choice([name_idx, gen.korutil.num2korord(name_idx)])

    num1_k = tokenpool.new(num1)
    name_idx_k = tokenpool.new(name_idx)
    subjects_k = list(map(tokenpool.new, subjects))

    question = '{place}에서 '
    question += f', '.join('{' + 'subjects{}'.format(x) + '}' for x in range(n))
    question += '의 {sent3}시험을 봤을 때, '
    question += f'{random.choice([name_idx, gen.korutil.num2korord(name_idx)])}번째로 '
    question += '{sent2}본 과목은 {sent1}과목{ques_trailing}'

    subject_dict = { f'subjects{i}': subjects_k[i] for i in range(n) }
    envdict = gen.fnmap(
                num1=num1_k,
                name_idx=name_idx_k,
                place=place,
                subjects=subjects_k,
                name=name,
                sent1=sent1,
                sent2=sent2,
                sent3=sent3,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            )
    envdict.update(subject_dict)

    return gen.build(
            body='',
            question=question,
            equation=gen.EqnRef('eqn02_1', subjects_k, name_idx_k),
            env=envdict
            )

# @gen.problems.register
def prob02_1_5(selector, tokenpool, clskey):
    ''' 계단에 정국, 태형, 지민이 순서대로 서 있습니다. 3번째에 서있는 사람은 누구입니까? '''

    # syntactic randomize
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent1 = random.choice(['순서로 ', '순으로 ', '순서대로 '])

    # new variables
    n=random.randint(2, 13)
    location = selector.get(clskey.location)
    order_lists=[selector.get(clskey.name) for _ in range(n)]
    order_idx = random.randint(1, len(order_lists))
    num1 = random.choice([order_idx, gen.korutil.num2korord(order_idx)])

    num1_k = tokenpool.new(num1)
    order_idx_k = tokenpool.new(order_idx)
    order_lists_k = list(map(tokenpool.new, order_lists))

    body = '{location}에 '
    body += f', '.join('{' + 'order_lists{}'.format(x) + '}' for x in range(n))
    body += '{#이?} {sent1}서 있습니다.'

    subject_dict = { f'order_lists{i}': order_lists_k[i] for i in range(n) }
    envdict = gen.fnmap(
                num1=num1_k,
                order_idx=order_idx_k,
                location=location,
                order_lists=order_lists_k,
                sent1=sent1,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            )
    envdict.update(subject_dict)

    return gen.build(
            body=body,
            question=f'{random.choice([order_idx, gen.korutil.num2korord(order_idx)])}번째로 '\
            '서있는 사람은 누구{ques_trailing}',
            equation=gen.EqnRef('eqn02_1', order_lists_k, order_idx_k),
            env=envdict
            )

# @gen.problems.register
def prob02_1_6(selector, tokenpool, clskey):
    ''' 서점, 마트, 문구점 순으로 넓을 때, 3번째로 넓은 장소(곳)은 어디입니까? '''

    # syntactic randomize
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent1 = random.choice(['순서로', '순으로', '순서대로'])
    sent2 = random.choice(['장소', '곳'])

    # new variables
    n=random.randint(2, 13)
    order_lists=[selector.get(clskey.place) for _ in range(n)]
    order_idx = random.randint(1, len(order_lists))
    num1 = random.choice([order_idx, gen.korutil.num2korord(order_idx)])

    num1_k = tokenpool.new(num1)
    order_idx_k = tokenpool.new(order_idx)
    order_lists_k = list(map(tokenpool.new, order_lists))

    question = f', '.join('{' + 'order_lists{}'.format(x) + '}' for x in range(n))
    question += ' {sent1} 넓을 때, '
    question += f'{num1}번째로 '
    question += '넓은 {sent2}{#는} 어디{ques_trailing}'

    subject_dict = { f'order_lists{i}': order_lists_k[i] for i in range(n) }
    envdict = gen.fnmap(
                num1=num1_k,
                order_idx=order_idx_k,
                order_lists=order_lists_k,
                sent1=sent1,
                sent2=sent2,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            )
    envdict.update(subject_dict)

    return gen.build(
            body='',
            question=question,
            equation=gen.EqnRef('eqn02_1', order_lists_k, order_idx_k),
            env=envdict
            )

# @gen.problems.register
def prob02_1_7(selector, tokenpool, clskey):
    ''' 달리기 시합에서 정국, 태형, 지민, ~~ 순서로 들어왔습니다. 
    3등(위)를 한 사람은 누구입니까? '''

    # syntactic randomize
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent1 = random.choice(['순서로', '순으로', '순서대로'])

    # new variables
    n=random.randint(2, 13)
    order_lists=[selector.get(clskey.name) for _ in range(n)]
    order_idx = random.randint(1, len(order_lists))
    num1 = random.choice([order_idx, gen.korutil.num2kor(order_idx)])
    rank = selector.get(clskey.rank)

    num1_k = tokenpool.new(num1)
    rank_k = tokenpool.new(rank)
    order_idx_k = tokenpool.new(order_idx)
    order_lists_k = list(map(tokenpool.new, order_lists))

    body = '달리기 시합에서 '
    body += f', '.join('{' + 'order_lists{}'.format(x) + '}' for x in range(n))
    body += '{#이?} {sent1} 들어왔습니다.'

    subject_dict = { f'order_lists{i}': order_lists_k[i] for i in range(n) }
    envdict = gen.fnmap(
                num1=num1_k,
                rank=rank_k,
                order_idx=order_idx_k,
                order_lists=order_lists_k,
                sent1=sent1,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            )
    envdict.update(subject_dict)

    return gen.build(
            body=body,
            question='{order_idx}{rank}{#를} 한 사람은 누구{ques_trailing}',
            equation=gen.EqnRef('eqn02_1', order_lists_k, order_idx_k),
            env=envdict
            )

# @gen.problems.register
def prob02_1_8(selector, tokenpool, clskey):
    ''' ord_rel = 크,무겁,빠르,높,많,길
    정국, 태형, 지민, ~~가 *키*가 작은 순서대로 줄을 섰습니다. 
    3번째로 키가 큰 사람은 누구입니까? '''

    # syntactic randomize
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent1 = random.choice(['순서로', '순으로', '순서대로'])

    # new variables
    n=random.randint(2, 7)

    exp1 = selector.get(clskey.ord_rel)
    if '크' in exp1.text:
        compared_obj = '키'
        sent2 = '줄을 섰습니다.'
        sent3 = '사람'
        sent4 = '누구'
        order_lists=[selector.get(clskey.name) for _ in range(n)]

    elif '무겁' in exp1.text:
        compared_obj = '몸무게'
        sent2 = '줄을 섰습니다.'
        sent3 = '사람'
        sent4 = '누구'
        order_lists=[selector.get(clskey.name) for _ in range(n)]

    elif '빠르' in exp1.text:
        compared_obj = '속도'
        sent2 = '들어왔습니다.'
        sent3 = '것'
        sent4 = '어느 것'
        order_lists=[selector.get(clskey.ride) for _ in range(n)]

    elif '높' in exp1.text:
        compared_obj = '높이'
        sent2 = ''

    elif '많' in exp1.text:
        compared_obj = '개수'
        sent2 = '있습니다.'
        sent3 = '것'
        sent4 = '어느 것'
        rand_num = random.randrange(3)
        if rand_num == 0:
            order_lists=[selector.get(clskey.fruit) for _ in range(n)] # 과일
        elif rand_num == 1:
            order_lists=[selector.get(clskey.name) for _ in range(n)] # 나이
            compared_obj = '나이'
            sent2 = '줄을 섰습니다.'
            sent3 = '사람'
            sent4 = '누구'
        elif rand_num == 2:
            order_lists=[selector.get(clskey.tool) for _ in range(n)] # 축구공

    elif '길' in exp1.text:
        compared_obj = '길이'
        sent2 = '있습니다.'
        sent3 = '것'
        sent4 = '어느 것'
        order_lists=[selector.get(clskey.wire) for _ in range(n)] # 선

    rand_num = random.randrange(2)
    if rand_num == 0:
            exp1 = exp1.of("adv")
    elif rand_num == 1:
            exp1 = exp1.of("reverse_adv")

    if sent3 == '사람':
        sent5 = '{#이?}'
    else:
        sent5 = ''

    order_idx = random.randint(1, len(order_lists))
    num1 = random.choice([order_idx, gen.korutil.num2korord(order_idx)])

    num1_k = tokenpool.new(num1)
    order_idx_k = tokenpool.new(order_idx)
    order_lists_k = list(map(tokenpool.new, order_lists))
    compared_obj_k = tokenpool.new(compared_obj)
    exp1_k = tokenpool.new(exp1)

    body = f', '.join('{' + 'order_lists{}'.format(x) + '}' for x in range(n))
    body += sent5 + '{#가} {compared_obj}{#가} {exp1} {sent1} {sent2}'

    subject_dict = { f'order_lists{i}': order_lists_k[i] for i in range(n) }
    envdict = gen.fnmap(
                num1=num1_k,
                compared_obj=compared_obj_k,
                order_idx=order_idx_k,
                order_lists=order_lists_k,
                sent1=sent1,
                sent2=sent2,
                sent3=sent3,
                sent4=sent4,
                sent5=sent5,
                exp1=exp1,

                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            )
    envdict.update(subject_dict)

    return gen.build(
            body=body,
            question=f'{random.choice([order_idx, gen.korutil.num2korord(order_idx)])}번째로 '\
            '{compared_obj}가 {exp1} {sent3}{#은} {sent4}{ques_trailing}',
            equation=gen.EqnRef('eqn02_1', order_lists_k, order_idx_k),
            env=envdict
            )

# @gen.problems.register
def prob02_1_9(selector, tokenpool, clskey):
    ''' ord_rel=크,무겁,빠르,높,많,길 --> 키 
    정국, 태형, 지민, ~~가 키가 작은 순서대로 줄을 *섰을 때, *
    3번째로 키가 큰 사람은 누구입니까? '''

    # syntactic randomize
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent1 = random.choice(['순서로', '순으로', '순서대로'])

    # new variables
    n=random.randint(2, 7)

    exp1 = selector.get(clskey.ord_rel)
    if '크' in exp1.text:
        compared_obj = '키'
        sent2 = '줄을 섰을 때, '
        sent3 = '사람'
        sent4 = '누구'
        order_lists=[selector.get(clskey.name) for _ in range(n)]

    elif '무겁' in exp1.text:
        compared_obj = '몸무게'
        sent2 = '줄을 섰을 때, '
        sent3 = '사람'
        sent4 = '누구'
        order_lists=[selector.get(clskey.name) for _ in range(n)]

    elif '빠르' in exp1.text:
        compared_obj = '속도'
        sent2 = '들어왔을 때, '
        sent3 = '것'
        sent4 = '어느 것'
        order_lists=[selector.get(clskey.ride) for _ in range(n)]

    elif '높' in exp1.text:
        compared_obj = '높이'
        sent2 = ''

    elif '많' in exp1.text:
        compared_obj = '개수'
        sent2 = '있을 때, '
        sent3 = '것'
        sent4 = '어느 것'
        rand_num = random.randrange(3)
        if rand_num == 0:
            order_lists=[selector.get(clskey.fruit) for _ in range(n)] # 과일
        elif rand_num == 1:
            order_lists=[selector.get(clskey.name) for _ in range(n)] # 나이
            compared_obj = '나이'
            sent2 = '줄을 섰을 때, '
            sent3 = '사람'
            sent4 = '누구'
        elif rand_num == 2:
            order_lists=[selector.get(clskey.tool) for _ in range(n)] # 축구공

    elif '길' in exp1.text:
        compared_obj = '길이'
        sent2 = '있을 때, '
        sent3 = '것'
        sent4 = '어느 것'
        order_lists=[selector.get(clskey.wire) for _ in range(n)] # 선

    rand_num = random.randrange(2)
    if rand_num == 0:
            exp1 = exp1.of("adv")
    elif rand_num == 1:
            exp1 = exp1.of("reverse_adv")

    if sent3 == '사람':
        sent5 = '{#이?}'
    else:
        sent5 = ''

    order_idx = random.randint(1, len(order_lists))
    num1 = random.choice([order_idx, gen.korutil.num2korord(order_idx)])

    num1_k = tokenpool.new(num1)
    order_idx_k = tokenpool.new(order_idx)
    order_lists_k = list(map(tokenpool.new, order_lists))
    compared_obj_k = tokenpool.new(compared_obj)
    exp1_k = tokenpool.new(exp1)

    question = f', '.join('{' + 'order_lists{}'.format(x) + '}' for x in range(n))
    question += sent5 + '{#가} {compared_obj}{#가} {exp1} {sent1} {sent2}'
    question += f'{random.choice([order_idx, gen.korutil.num2korord(order_idx)])}번째로 '
    question += '{compared_obj}가 {exp1} {sent3}{#은} {sent4}{ques_trailing}'

    subject_dict = { f'order_lists{i}': order_lists_k[i] for i in range(n) }
    envdict = gen.fnmap(
                num1=num1_k,
                compared_obj=compared_obj_k,
                order_idx=order_idx_k,
                order_lists=order_lists_k,
                sent1=sent1,
                sent2=sent2,
                sent3=sent3,
                sent4=sent4,
                sent5=sent5,
                exp1=exp1,

                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            )
    envdict.update(subject_dict)

    return gen.build(
            body='',
            question=question,
            equation=gen.EqnRef('eqn02_1', order_lists_k, order_idx_k),
            env=envdict
            )



######################################################################################
################################## problem 02 ########################################

# @gen.problems.register
def prob02_2_1(selector, tokenpool, clskey):
    ''' 달리기 시합에서 정국이는 7등을 했고, 민영이는 5등을 했습니다. 태형이는 민영이보다
    못했지만 정국이보다는 잘했습니다.  
    태형이는 몇 등입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])

    # new variables
    race = selector.get(clskey.race)
    rank_name = selector.get(clskey.rank)

    max_rank = 100
    rank1 = random.randint(3, max_rank) # 7등
    rank2 = rank1 - 2                   # 5등

    name1 = selector.get(clskey.name) # 정국
    name2 = selector.get(clskey.name) # 민영
    name3 = selector.get(clskey.name) # 태형

    rank1_k = tokenpool.new(rank1)
    rank2_k = tokenpool.new(rank2)
    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    name3_k = tokenpool.new(name3)

    return gen.build(
            body=' '.join([
                '{race}에서 {name1}{#이?}는 {rank1}{rank_name}{#을} 했고, '\
                '{name2}{#이?}는 {rank2}{rank_name}{#을} 했습니다. '\
                '{name3}{#이?}는 {name2}{#이?}보다 못했지만 '\
                '{name1}{#이?}보다는 잘했{sent_trailing}'
            ]),
            question='{name3}{#이?}는 몇 {rank_name}{ques_trailing}',
            equation=gen.EqnRef('eqn02_2', rank2_k, rank1_k),

            env=gen.fnmap(
                race=race,
                rank_name=rank_name,
                rank1=rank1_k,
                rank2=rank2_k,
                name1=name1_k,
                name2=name2_k,
                name3=name3_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_2_2(selector, tokenpool, clskey):
    ''' 달리기 시합에서 민영이는 5등을 했고, 정국이는 7등을 했습니다. 태형이는 민영이보다
    못했지만 정국이보다는 잘했습니다.  
    태형이는 몇 등입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])

    # new variables
    race = selector.get(clskey.race)
    rank_name = selector.get(clskey.rank)

    max_rank = 100
    rank1 = random.randint(3, max_rank) # 7등
    rank2 = rank1 - 2                   # 5등

    name1 = selector.get(clskey.name) # 정국
    name2 = selector.get(clskey.name) # 민영
    name3 = selector.get(clskey.name) # 태형

    rank1_k = tokenpool.new(rank1)
    rank2_k = tokenpool.new(rank2)
    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    name3_k = tokenpool.new(name3)

    return gen.build(
            body=' '.join([
                '{race}에서 {name2}{#이?}는 {rank2}{rank_name}{#을} 했고, '\
                '{name1}{#이?}는 {rank1}{rank_name}{#을} 했습니다. '\
                '{name3}{#이?}는 {name2}{#이?}보다 못했지만 '\
                '{name1}{#이?}보다는 잘했{sent_trailing}'
            ]),
            question='{name3}{#이?}는 몇 {rank_name}{ques_trailing}',
            equation=gen.EqnRef('eqn02_2', rank2_k, rank1_k),

            env=gen.fnmap(
                race=race,
                rank_name=rank_name,
                rank1=rank1_k,
                rank2=rank2_k,
                name1=name1_k,
                name2=name2_k,
                name3=name3_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_2_3(selector, tokenpool, clskey):
    ''' 달리기 시합에서 민영이는 5등을 했고, 정국이는 7등을 했습니다. 
    태형이는 정국이보다는 잘했지만 민영이보다 못했습니다.  
    태형이는 몇 등입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])

    # new variables
    race = selector.get(clskey.race)
    rank_name = selector.get(clskey.rank)

    max_rank = 100
    rank1 = random.randint(3, max_rank) # 7등
    rank2 = rank1 - 2                   # 5등

    name1 = selector.get(clskey.name) # 정국
    name2 = selector.get(clskey.name) # 민영
    name3 = selector.get(clskey.name) # 태형

    rank1_k = tokenpool.new(rank1)
    rank2_k = tokenpool.new(rank2)
    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    name3_k = tokenpool.new(name3)

    return gen.build(
            body=' '.join([
                '{race}에서 {name2}{#이?}는 {rank2}{rank_name}{#을} 했고, '\
                '{name1}{#이?}는 {rank1}{rank_name}{#을} 했습니다. '\
                '{name3}{#이?}는 {name1}{#이?}보다는 잘했지만 '\
                '{name2}{#이?}보다 못했{sent_trailing}'
            ]),
            question='{name3}{#이?}는 몇 {rank_name}{ques_trailing}',
            equation=gen.EqnRef('eqn02_2', rank2_k, rank1_k),

            env=gen.fnmap(
                race=race,
                rank_name=rank_name,
                rank1=rank1_k,
                rank2=rank2_k,
                name1=name1_k,
                name2=name2_k,
                name3=name3_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_2_4(selector, tokenpool, clskey):
    ''' 민영이와 정국이가 달리기 시합에서 각각 5등, 7등을 했습니다.(했고, )
    태형이는 정국이보다는 잘했지만 민영이보다 못했습니다.
    태형이는 몇 등을 했는지 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['했습니까?', '했는지 구하시오.', '했는지 구하세요.'])
    sent1 = random.choice([' ', '는 '])
    sent2 = random.choice(['했습니다. ', '했고, '])

    # new variables
    race = selector.get(clskey.race)
    rank_name = selector.get(clskey.rank)

    max_rank = 100
    rank1_ = random.randint(3, max_rank) # 7등
    rank2_ = rank1_ - 2                   # 5등

    name1_ = selector.get(clskey.name) # 정국
    name2_ = selector.get(clskey.name) # 민영
    name3 = selector.get(clskey.name) # 태형

    rank1 = random.choice([rank1_, rank2_])
    rand_num = random.randrange(5)
    speed_exp = [
        ['빨랐', '느렸'],  ['잘했', '못했'], ['빨리 들어왔', '뒤에 들어왔'],
        ['먼저 들어왔', '늦게 들어왔'], ['일찍 들어왔', '늦게 들어왔']
    ]
    if rank1 == rank1_:
        rank2 = rank2_
        name1 = name1_
        name2 = name2_
        speed_1 = speed_exp[rand_num][0]
        speed_2 = speed_exp[rand_num][1]
    else:
        rank2 = rank1_
        name1 = name2_
        name2 = name1_
        speed_1 = speed_exp[rand_num][1]
        speed_2 = speed_exp[rand_num][0]

    rank1_k = tokenpool.new(rank1)
    rank2_k = tokenpool.new(rank2)
    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    name3_k = tokenpool.new(name3)

    return gen.build(
            body=' '.join([
                '{name2}{#이?}와 {name1}{#이?}가 {race}에서 각각 {rank2}{rank_name}, '\
                '{rank1}{rank_name}{#을} {sent2}'\
                '{name3}{#이?}는 {name1}{#이?}보다{sent1}{speed_1}지만 '\
                '{name2}{#이?}보다{sent1}{speed_2}{sent_trailing}'
            ]),
            question='{name3}{#이?}는 몇 {rank_name}{#을} {ques_trailing}',
            equation=gen.EqnRef('eqn02_2', rank2_k, rank1_k),

            env=gen.fnmap(
                race=race,
                rank_name=rank_name,
                rank1=rank1_k,
                rank2=rank2_k,
                name1=name1_k,
                name2=name2_k,
                name3=name3_k,
                speed_1=speed_1,
                speed_2=speed_2,
                sent1=sent1,
                sent2=sent2,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))


# @gen.problems.register
def prob02_2_5(selector, tokenpool, clskey):
    ''' 달리기 시합에서 민영이는 5등을 했고, 정국이는 7등을 했습니다. 
    태형이는 정국이보다는 *잘했*지만 민영이보다 *못했*습니다.  
    태형이는 몇 등입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['했습니까?', '했는지 구하시오.', '했는지 구하세요.'])
    sent1 = random.choice([' ', '는 '])
    sent2 = random.choice(['했습니다. ', '했고, '])
    sent3 = random.choice(['습니다. ', '을 때, '])
    sent4 = random.choice([' ', '는 '])

    # new variables
    race = selector.get(clskey.race)
    rank_name = selector.get(clskey.rank)

    max_rank = 100
    rank1_ = random.randint(3, max_rank) # 7등
    rank2_ = rank1_ - 2                   # 5등

    name1_ = selector.get(clskey.name) # 정국
    name2_ = selector.get(clskey.name) # 민영
    name3 = selector.get(clskey.name) # 태형

    rank1 = random.choice([rank1_, rank2_])
    rand_num = random.randrange(5)
    speed_exp = [
        ['빨랐', '느렸'],  ['잘했', '못했'], ['빨리 들어왔', '뒤에 들어왔'],
        ['먼저 들어왔', '늦게 들어왔'], ['일찍 들어왔', '늦게 들어왔']
    ]
    if rank1 == rank1_:
        rank2 = rank2_
        name1 = name1_
        name2 = name2_
        speed_1 = speed_exp[rand_num][0]
        speed_2 = speed_exp[rand_num][1]
    else:
        rank2 = rank1_
        name1 = name2_
        name2 = name1_
        speed_1 = speed_exp[rand_num][1]
        speed_2 = speed_exp[rand_num][0]

    rank1_k = tokenpool.new(rank1)
    rank2_k = tokenpool.new(rank2)
    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    name3_k = tokenpool.new(name3)

    return gen.build(
            body=' '.join([
                '{race}에서 {name1}{#이?}는 {rank1}{rank_name}{#을} 했고, '\
                '{name2}{#이?}는 {rank2}{rank_name}{#을} 했{sent3}'\
                '{name3}{#이?}는 {name2}{#이?}보다{sent1}{speed_2}지만 '\
                '{name1}{#이?}보다{sent4}{speed_1}{sent_trailing}'
            ]),
            question='{name3}{#이?}는 몇 {rank_name}{#을} {ques_trailing}',
            equation=gen.EqnRef('eqn02_2', rank2_k, rank1_k),

            env=gen.fnmap(
                race=race,
                rank_name=rank_name,
                rank1=rank1_k,
                rank2=rank2_k,
                name1=name1_k,
                name2=name2_k,
                name3=name3_k,
                speed_1=speed_1,
                speed_2=speed_2,
                sent1=sent1,
                sent2=sent2,
                sent3=sent3,
                sent4=sent4,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))


# @gen.problems.register
def prob02_2_6(selector, tokenpool, clskey):
    ''' 달리기 시합에서 민영이는 5등을 했고, 정국이는 7등을 했습니다. 
    태형이는 지만 민영이보다 *못했*정국이보다*는* *잘했*습니다.  
    태형이는 몇 등입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['했습니까?', '했는지 구하시오.', '했는지 구하세요.'])
    sent1 = random.choice([' ', '는 '])
    sent2 = random.choice(['했습니다. ', '했고, '])
    sent3 = random.choice(['습니다. ', '을 때, '])
    sent4 = random.choice([' ', '는 '])

    # new variables
    race = selector.get(clskey.race)
    rank_name = selector.get(clskey.rank)

    max_rank = 100
    rank1_ = random.randint(3, max_rank) # 7등
    rank2_ = rank1_ - 2                   # 5등

    name1_ = selector.get(clskey.name) # 정국
    name2_ = selector.get(clskey.name) # 민영
    name3 = selector.get(clskey.name) # 태형

    rank1 = random.choice([rank1_, rank2_])
    rand_num = random.randrange(5)
    speed_exp = [
        ['빨랐', '느렸'],  ['잘했', '못했'], ['빨리 들어왔', '뒤에 들어왔'],
        ['먼저 들어왔', '늦게 들어왔'], ['일찍 들어왔', '늦게 들어왔']
    ]
    if rank1 == rank1_:
        rank2 = rank2_
        name1 = name1_
        name2 = name2_
        speed_1 = speed_exp[rand_num][0]
        speed_2 = speed_exp[rand_num][1]
    else:
        rank2 = rank1_
        name1 = name2_
        name2 = name1_
        speed_1 = speed_exp[rand_num][1]
        speed_2 = speed_exp[rand_num][0]

    rank1_k = tokenpool.new(rank1)
    rank2_k = tokenpool.new(rank2)
    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    name3_k = tokenpool.new(name3)

    return gen.build(
            body=' '.join([
                '{race}에서 {name2}{#이?}는 {rank2}{rank_name}{#을} 했고, '\
                '{name1}{#이?}는 {rank1}{rank_name}{#을} 했{sent3}'\
                '{name3}{#이?}는 {name1}{#이?}보다{sent1}{speed_1}지만 '\
                '{name2}{#이?}보다{sent4}{speed_2}{sent_trailing}'
            ]),


            question='{name3}{#이?}는 몇 {rank_name}{#을} {ques_trailing}',
            equation=gen.EqnRef('eqn02_2', rank2_k, rank1_k),

            env=gen.fnmap(
                race=race,
                rank_name=rank_name,
                rank1=rank1_k,
                rank2=rank2_k,
                name1=name1_k,
                name2=name2_k,
                name3=name3_k,
                speed_1=speed_1,
                speed_2=speed_2,
                sent1=sent1,
                sent2=sent2,
                sent3=sent3,
                sent4=sent4,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))


# @gen.problems.register
def prob02_2_7(selector, tokenpool, clskey):
    ''' 달리기 시합에서 민영이는 5등을 했고, 정국이는 7등을 했습니다. 
    태형이는 지만 민영이보다 *점수가 낮았*지만 정국이보다*는* *높았*습니다.  
    태형이는 몇 등입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['했습니까?', '했는지 구하시오.', '했는지 구하세요.'])
    sent1 = random.choice([' ', '는 '])
    sent2 = random.choice(['했습니다. ', '했고, '])
    sent3 = random.choice(['습니다. ', '을 때, '])
    sent4 = random.choice([' ', '는 '])

    # new variables
    race = selector.get(clskey.race)
    rank_name = selector.get(clskey.rank)

    max_rank = 100
    rank1 = random.randint(3, max_rank) # 7등
    rank2 = rank1 - 2                   # 5등

    name1_ = selector.get(clskey.name) # 정국
    name2_ = selector.get(clskey.name) # 민영
    name3 = selector.get(clskey.name) # 태형

    speed_1 = random.choice(['빨랐', '잘했', '빨리 들어왔', '먼저 들어왔', '일찍 들어왔'])
    speed_2 = random.choice(['느렸', '못했', '뒤에 들어왔', '늦게 들어왔'])
    if '빨' or '잘' or '먼저' or '일찍' in speed_1:
        name1 = name1_
        name2 = name2_
    else:
        name1 = name2_
        name2 = name1_

    rank1_k = tokenpool.new(rank1)
    rank2_k = tokenpool.new(rank2)
    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    name3_k = tokenpool.new(name3)

    return gen.build(
            body=' '.join([
                '달리기 시합에서 {name2}{#이?}는 {rank2}{rank_name}{#을} 했고, '\
                '{name1}{#이?}는 {rank1}{rank_name}{#을} 했{sent3}'\
                '{name3}{#이?}는 {name1}{#이?}보다{sent1} 점수가 높았지만 '\
                '{name2}{#이?}보다{sent4}낮았{sent_trailing}'
            ]),
            question='{name3}{#이?}는 몇 {rank_name}{#을} {ques_trailing}',
            equation=gen.EqnRef('eqn02_2', rank2_k, rank1_k),

            env=gen.fnmap(
                race=race,
                rank_name=rank_name,
                rank1=rank1_k,
                rank2=rank2_k,
                name1=name1_k,
                name2=name2_k,
                name3=name3_k,
                speed_1=speed_1,
                speed_2=speed_2,
                sent1=sent1,
                sent2=sent2,
                sent3=sent3,
                sent4=sent4,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_2_8(selector, tokenpool, clskey):
    ''' 민영이와 정국이 사이에 태형이가 있습니다. 
    민영이는 5번째에 있고, 정국이는 7번째에 있다면 태형이는 몇 번째에 있습니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['습니까?', '는지 구하시오.', '는지 구하세요.', '는지 구해 보세요.', '는가?'])

    # new variables
    max_rank = 100
    rank1 = random.randint(3, max_rank) # 7등
    rank2 = rank1 - 2                   # 5등

    name1 = selector.get(clskey.name) # 정국
    name2 = selector.get(clskey.name) # 민영
    name3 = selector.get(clskey.name) # 태형

    rand_num = random.randrange(2)
    num1 = [rank1, gen.korutil.num2korord(rank1)][rand_num]
    num2 = [rank2, gen.korutil.num2korord(rank2)][rand_num]

    rank1_k = tokenpool.new(rank1)
    rank2_k = tokenpool.new(rank2)
    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    name3_k = tokenpool.new(name3)

    return gen.build(
            body='{name2}{#이?}와 {name1}{#이?} 사이에 {name3}{#이?}가 있습니다.',
            question='{name2}{#이?}는 '\
            f'{[rank1, gen.korutil.num2korord(rank1)][rand_num]}번째에 있고, '\
            '{name1}{#이?}는 '\
            f'{[rank2, gen.korutil.num2korord(rank2)][rand_num]}번째에 있다면 '\
            '{name3}{#이?}는 몇 번째에 있{ques_trailing}',
            equation=gen.EqnRef('eqn02_2', rank2_k, rank1_k),

            env=gen.fnmap(
                rank1=rank1_k,
                rank2=rank2_k,
                name1=name1_k,
                name2=name2_k,
                name3=name3_k,

                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_2_9(selector, tokenpool, clskey):
    ''' 달리기 시합에서 정국이는 7번째로 들어왔고, 민영이는 5번째로 들어왔습니다. 
    태형이가 민영이보다 늦게 들어왔지만, 정국이보다 빨리 들어왔을 때, 태형이는 몇 등입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['했습니까?', '했는지 구하시오.', '했는지 구하세요.'])
    sent1 = random.choice([' ', '는 '])
    sent2 = random.choice(['했습니다. ', '했고, '])
    sent3 = random.choice(['습니다. ', '을 때, '])
    sent4 = random.choice([' ', '는 '])

    # new variables
    race = selector.get(clskey.race)
    rank_name = selector.get(clskey.rank)

    max_rank = 100
    rank1 = random.randint(3, max_rank) # 7등
    rank2 = rank1 - 2                   # 5등

    name1_ = selector.get(clskey.name) # 정국
    name2_ = selector.get(clskey.name) # 민영
    name3 = selector.get(clskey.name) # 태형

    rand_num = random.randrange(2)
    num1 = [rank1, gen.korutil.num2korord(rank1)][rand_num]
    num2 = [rank2, gen.korutil.num2korord(rank2)][rand_num]

    speed_1 = random.choice(['빨랐', '잘했', '빨리 들어왔', '먼저 들어왔', '일찍 들어왔'])
    speed_2 = random.choice(['느렸', '못했', '뒤에 들어왔', '늦게 들어왔'])
    if '빨' or '잘' or '먼저' or '일찍' in speed_1:
        name1 = name1_
        name2 = name2_
    else:
        name1 = name2_
        name2 = name1_

    rank1_k = tokenpool.new(rank1)
    rank2_k = tokenpool.new(rank2)
    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    name3_k = tokenpool.new(name3)

    return gen.build(
            body='달리기 시합에서 {name2}{#이?}는 '\
            f'{[rank1, gen.korutil.num2korord(rank1)][rand_num]}번째에 들어왔고, '\
            '{name1}{#이?}는 '\
            f'{[rank2, gen.korutil.num2korord(rank2)][rand_num]}번째에 들어왔습니다.',
            question='{name3}{#이?}는 {name1}{#이?}보다{sent1}{speed_2}지만 '\
            '{name2}{#이?}보다{sent4}{speed_1}을 때, '\
            '{name3}{#이?}는 몇 {rank_name}{#을} {ques_trailing}',
            equation=gen.EqnRef('eqn02_2', rank2_k, rank1_k),

            env=gen.fnmap(
                race=race,
                rank_name=rank_name,
                rank1=rank1_k,
                rank2=rank2_k,
                name1=name1_k,
                name2=name2_k,
                name3=name3_k,
                speed_1=speed_1,
                speed_2=speed_2,
                sent1=sent1,
                sent2=sent2,
                sent3=sent3,
                sent4=sent4,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_2_10(selector, tokenpool, clskey):
    ''' 정국이와 민영이와 태형이가 달리기 시합에 참가하였습니다. 
    정국이와 민영이는 각각 7등, 5등을 하였습니다. 태형이는 민영이보다는 낮은 점수를 받았지만, 정국이보다는 높은 점수를 받았습니다. 
    태형이는 몇 등입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['했습니까?', '했는지 구하시오.', '했는지 구하세요.'])
    sent1 = random.choice([' ', '는 '])
    sent2 = random.choice(['했습니다. ', '했고, '])
    sent3 = random.choice(['습니다. ', '을 때, '])
    sent4 = random.choice([' ', '는 '])

    # new variables
    race = selector.get(clskey.race)
    rank_name = selector.get(clskey.rank)

    max_rank = 100
    rank1_ = random.randint(3, max_rank) # 7등
    rank2_ = rank1_ - 2                   # 5등

    name1_ = selector.get(clskey.name) # 정국
    name2_ = selector.get(clskey.name) # 민영
    name3 = selector.get(clskey.name) # 태형

    rank1 = random.choice([rank1_, rank2_])
    rand_num = random.randrange(5)
    speed_exp = ['높은', '낮은']
    if rank1 == rank1_:
        rank2 = rank2_
        name1 = name1_
        name2 = name2_
        speed_1 = speed_exp[0]
        speed_2 = speed_exp[1]
    else:
        rank2 = rank1_
        name1 = name2_
        name2 = name1_
        speed_1 = speed_exp[1]
        speed_2 = speed_exp[0]

    rank1_k = tokenpool.new(rank1)
    rank2_k = tokenpool.new(rank2)
    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    name3_k = tokenpool.new(name3)

    return gen.build(
            body='{name1}{#이?}와 {name2}{#이?}와 {name3}{#이?}가 {race}에 참가하였습니다. '\
            '{name1}{#이?}와 {name2}{#이?}는 각각 {rank1}{rank_name}, {rank2}{rank_name}을 하였습니다. '\
            '{name3}{#이?}는 {name1}{#이?}보다{sent1}{speed_1} 점수를 받았지만, '\
            '{name2}{#이?}보다{sent4}{speed_2} 점수를 받았습니다.',
            question='{name3}{#이?}는 몇 {rank_name}{#을} {ques_trailing}',
            equation=gen.EqnRef('eqn02_2', rank2_k, rank1_k),

            env=gen.fnmap(
                race=race,
                rank_name=rank_name,
                rank1=rank1_k,
                rank2=rank2_k,
                name1=name1_k,
                name2=name2_k,
                name3=name3_k,
                speed_1=speed_1,
                speed_2=speed_2,
                sent1=sent1,
                sent2=sent2,
                sent3=sent3,
                sent4=sent4,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))


######################################################################################
################################## problem 03 ########################################

# @gen.problems.register
def prob02_3_1(selector, tokenpool, clskey):
    ''' 학생들이 한 줄로 서 있습니다. 유정이는 맨 뒤에 서 있습니다. 은정이는 앞에서
    5번째에 서 있습니다. 
    은정이와 유정이 사이에 8명이 서 있을 때, 줄을 서 있는 학생은 모두 몇 명입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    line = random.choice(['', '줄을', '한 줄로 줄을', '일렬로', '한 줄로'])
    sent1 = random.choice(['습니다. ', '고, '])
    sent2_choice = ['는', '가']
    sent3 = random.choice(['', '들'])
    sent4 = random.choice(['', '만약 '])
    sent5 = random.choice(['이 ', '의 사람들이 '])

    # new variables
    person = selector.get(clskey.person)
    rank_name = selector.get(clskey.rank)
    name1 = selector.get(clskey.name)
    name2 = selector.get(clskey.name)

    max_idx = 100
    order1 = random.randint(1, max_idx-2)
    order2 = random.randint(order1+2, max_idx)
    n_people_btw = random.randint(1, order2-order1-1)
    random.shuffle(sent2_choice)
    sent2_1 = sent2_choice[0]
    sent2_2 = sent2_choice[1]

    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    order1_k = tokenpool.new(order1)
    n_people_btw_k = tokenpool.new(n_people_btw)

    return gen.build(
            body=' '.join([
                '{person}들이 {line} 서 있{sent1}{name1}{#이?}{sent2_1} 맨 뒤에 서 있{sent1}'\
                '{name2}{#이?}{sent2_2} 앞에서 {order1}번째에 서 있습니다.'
            ]),
            question='{sent4}{name2}{#이?}와 {name1}{#이?} 사이에 {n_people_btw}명{sent5} 서 있을 때, '\
            '줄을 서 있는 {person}{sent3}{#은} {total}몇 명{ques_trailing}',
            equation=gen.EqnRef('eqn02_3', order1_k, n_people_btw_k),

            env=gen.fnmap(
                person=person,
                rank_name=rank_name,
                line=line,
                sent1=sent1,
                sent2_1=sent2_1,
                sent2_2=sent2_2,
                sent3=sent3,
                sent4=sent4,
                sent5=sent5,
                order1=order1_k,
                n_people_btw=n_people_btw_k,
                name1=name1_k,
                name2=name2_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_3_2(selector, tokenpool, clskey):
    ''' 학생들이 한 줄로 서 있습니다. 유정이는 맨 뒤에 서 있습니다. 은정이는 앞에서
    5번째에 서 있습니다. 
    은정이와 유정이 사이에 8명이 서 있을 때, *모두 몇 명의 학생들이 줄을 서 있는지 구하시오.* '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['습니까?', '는지 구하시오.', '는지 구하세요.', '는지 구해 보세요.', '는가?'])
    line = random.choice(['', '줄을', '한 줄로 줄을', '일렬로', '한 줄로'])
    sent1 = random.choice(['습니다. ', '고, '])
    sent2_choice = ['는', '가']
    sent3 = random.choice(['', '들'])
    sent4 = random.choice(['', '만약 '])
    sent5 = random.choice(['이 ', '의 사람들이 '])

    # new variables
    person = selector.get(clskey.person)
    rank_name = selector.get(clskey.rank)
    name1 = selector.get(clskey.name)
    name2 = selector.get(clskey.name)

    max_idx = 100
    order1 = random.randint(1, max_idx-2)
    order2 = random.randint(order1+2, max_idx)
    n_people_btw = random.randint(1, order2-order1-1)
    random.shuffle(sent2_choice)
    sent2_1 = sent2_choice[0]
    sent2_2 = sent2_choice[1]

    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    order1_k = tokenpool.new(order1)
    n_people_btw_k = tokenpool.new(n_people_btw)

    return gen.build(
            body=' '.join([
                '{person}들이 {line} 서 있{sent1}{name1}{#이?}{sent2_1} 맨 뒤에 서 있{sent1}'\
                '{name2}{#이?}{sent2_2} 앞에서 {order1}번째에 서 있습니다.'
            ]),
            question='{sent4}{name2}{#이?}와 {name1}{#이?} 사이에 {n_people_btw}명{sent5} 서 있을 때, '\
            '{total}몇 명의 {person}{sent3}이 줄을 서 있{ques_trailing}',
            equation=gen.EqnRef('eqn02_3', order1_k, n_people_btw_k),

            env=gen.fnmap(
                person=person,
                rank_name=rank_name,
                line=line,
                sent1=sent1,
                sent2_1=sent2_1,
                sent2_2=sent2_2,
                sent3=sent3,
                sent4=sent4,
                sent5=sent5,
                order1=order1_k,
                n_people_btw=n_people_btw_k,
                name1=name1_k,
                name2=name2_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_3_3(selector, tokenpool, clskey):
    ''' 학생들이 한 줄로 서 있습니다. *은정이가 앞에서 5번째에 서 있고, 
    유정이는 맨 뒤에 서 있습니다.* 
    은정이와 유정이 사이에 8명이 서 있을 때, 모두 몇 명의 학생들이 줄을 서 있는지 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['습니까?', '는지 구하시오.', '는지 구하세요.', '는지 구해 보세요.', '는가?'])
    line = random.choice(['', '줄을', '한 줄로 줄을', '일렬로', '한 줄로'])
    sent1 = random.choice(['습니다. ', '고, '])
    sent2_choice = ['는', '가']
    sent3 = random.choice(['', '들'])
    sent4 = random.choice(['', '만약 '])
    sent5 = random.choice(['이 ', '의 사람들이 '])

    # new variables
    person = selector.get(clskey.person)
    rank_name = selector.get(clskey.rank)
    name1 = selector.get(clskey.name)
    name2 = selector.get(clskey.name)

    max_idx = 100
    order1 = random.randint(1, max_idx-2)
    order2 = random.randint(order1+2, max_idx)
    n_people_btw = random.randint(1, order2-order1-1)
    random.shuffle(sent2_choice)
    sent2_1 = sent2_choice[0]
    sent2_2 = sent2_choice[1]

    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    order1_k = tokenpool.new(order1)
    n_people_btw_k = tokenpool.new(n_people_btw)

    return gen.build(
            body='{person}들이 {line} 서 있{sent1}{name2}{#이?}{sent2_2} 앞에서 {order1}번째에 서 있{sent1}'\
            '{name1}{#이?}{sent2_1} 맨 뒤에 서 있습니다.',
            question='{sent4}{name2}{#이?}와 {name1}{#이?} 사이에 {n_people_btw}명{sent5} 서 있을 때, '\
            '{total}몇 명의 {person}{sent3}이 줄을 서 있{ques_trailing}',
            equation=gen.EqnRef('eqn02_3', order1_k, n_people_btw_k),

            env=gen.fnmap(
                person=person,
                rank_name=rank_name,
                line=line,
                sent1=sent1,
                sent2_1=sent2_1,
                sent2_2=sent2_2,
                sent3=sent3,
                sent4=sent4,
                sent5=sent5,
                order1=order1_k,
                n_people_btw=n_people_btw_k,
                name1=name1_k,
                name2=name2_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_3_4(selector, tokenpool, clskey):
    ''' 유정이가 맨 뒤, 은정이가 5번째에 줄을 서 있습니다.
    둘 사이에 8명이 서 있을 때, 줄을 서 있는 학생은 모두 몇 명입니까?* '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    line = random.choice(['', '줄을', '한 줄로 줄을', '일렬로', '한 줄로'])
    sent1 = random.choice(['을 때, ', '있다면, '])
    sent2_choice = ['는', '가']
    sent3 = random.choice(['', '들'])
    sent5 = random.choice(['이 ', '의 사람들이 '])


    # new variables
    person = selector.get(clskey.person)
    rank_name = selector.get(clskey.rank)
    name1 = selector.get(clskey.name) # 유정
    name2 = selector.get(clskey.name) # 은정

    max_idx = 100
    order1 = random.randint(1, max_idx-2) # 5번째
    order2 = random.randint(order1+2, max_idx)
    n_people_btw = random.randint(1, order2-order1-1) # 8명
    random.shuffle(sent2_choice)
    sent2_1 = sent2_choice[0]
    sent2_2 = sent2_choice[1]

    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    order1_k = tokenpool.new(order1)
    n_people_btw_k = tokenpool.new(n_people_btw)

    return gen.build(
            body='{name1}{#이?}{sent2_1} 맨 뒤, {name2}{#이?}{sent2_2} 앞에서 {order1}번째에 서 있습니다.',
            question='둘 사이에 {n_people_btw}명{sent5} 서 있{sent1}줄을 서 있는 {person}{sent3}{#은} '\
                '{total}몇 명{ques_trailing}',
            equation=gen.EqnRef('eqn02_3', order1_k, n_people_btw_k),

            env=gen.fnmap(
                person=person,
                rank_name=rank_name,
                line=line,
                sent1=sent1,
                sent2_1=sent2_1,
                sent2_2=sent2_2,
                sent3=sent3,
                sent5=sent5,
                order1=order1_k,
                n_people_btw=n_people_btw_k,
                name1=name1_k,
                name2=name2_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_3_5(selector, tokenpool, clskey):
    ''' 유정이와 은정이가 줄을 섰는데, 유정이는 맨 뒤에 서 있고, 은정이는 앞에서 5번째에 서 있습니다. 
    은정이와 유정이 사이에 8명이 서 있다면, 줄을 서 있는 학생은 모두 몇 명입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    line = random.choice(['', '줄을', '한 줄로 줄을', '일렬로', '한 줄로'])
    sent1 = random.choice(['을 때, ', '있다면, '])
    sent2_choice = ['는', '가']
    sent3 = random.choice(['', '들'])
    sent5 = random.choice(['이 ', '의 사람들이 '])
    sent6 = random.choice(['맨 ', '가장 '])


    # new variables
    person = selector.get(clskey.person)
    rank_name = selector.get(clskey.rank)
    name1 = selector.get(clskey.name) # 유정
    name2 = selector.get(clskey.name) # 은정

    max_idx = 100
    order1 = random.randint(1, max_idx-2) # 5번째
    order2 = random.randint(order1+2, max_idx)
    n_people_btw = random.randint(1, order2-order1-1) # 8명
    random.shuffle(sent2_choice)
    sent2_1 = sent2_choice[0]
    sent2_2 = sent2_choice[1]

    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    order1_k = tokenpool.new(order1)
    n_people_btw_k = tokenpool.new(n_people_btw)

    return gen.build(
            body='{name1}{#이?}와 {name2}{#이?}가 줄을 섰는데, {name1}{#이?}{#는} {sent6}뒤에 서 있고, '\
            '{name2}{#이?}는 앞에서 {order1}번째에 서 있습니다.',
            question='{name1}{#이?}{sent2_1} {sent6}뒤, {name2}{#이?}{sent2_2} 앞에서 {order1}번째에 서 있'\
                '{name1}{#이?}와 {name2}{#이?} 사이에 {n_people_btw}명{sent5} 서 있{sent1}줄을 서 있는 {person}{sent3}{#은} '\
                '{total}몇 명{ques_trailing}',
            equation=gen.EqnRef('eqn02_3', order1_k, n_people_btw_k),

            env=gen.fnmap(
                person=person,
                rank_name=rank_name,
                line=line,
                sent1=sent1,
                sent2_1=sent2_1,
                sent2_2=sent2_2,
                sent3=sent3,
                sent5=sent5,
                sent6=sent6,
                order1=order1_k,
                n_people_btw=n_people_btw_k,
                name1=name1_k,
                name2=name2_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_3_6(selector, tokenpool, clskey):
    ''' 유정이는 가장 마지막에 매장에 들어왔고, 은정이는 5번째로 들어왔다. 
    유정이와 은정이 사이에 8명이 들어왔을 때, 매장에 들어온 사람은 모두 몇 명입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    line = random.choice(['', '줄을', '한 줄로 줄을', '일렬로', '한 줄로'])
    sent1 = random.choice(['을 때, ', '있다면, '])
    sent2_choice = ['는', '가']
    sent3 = random.choice(['', '들'])
    sent5 = random.choice(['이 ', '의 사람들이 '])
    sent6 = random.choice(['맨 ', '가장 '])

    # new variables
    person = selector.get(clskey.person)
    rank_name = selector.get(clskey.rank)
    name1 = selector.get(clskey.name) # 유정
    name2 = selector.get(clskey.name) # 은정
    place = selector.get(clskey.place)

    max_idx = 100
    order1 = random.randint(1, max_idx-2) # 5번째
    order2 = random.randint(order1+2, max_idx)
    n_people_btw = random.randint(1, order2-order1-1) # 8명
    random.shuffle(sent2_choice)
    sent2_1 = sent2_choice[0]
    sent2_2 = sent2_choice[1]

    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    order1_k = tokenpool.new(order1)
    n_people_btw_k = tokenpool.new(n_people_btw)

    return gen.build(
            body='{name1}{#이?}는 {sent6}마지막에 {place}에 들어왔고, {name2}{#이?}는 '\
            f'{random.choice([order1, gen.korutil.num2korord(order1)])}번째로 들어왔다.',
            question='{name1}{#이?}와 {name2}{#이?} 사이에 {n_people_btw}명{sent5}들어왔을 때, '\
            '{place}에 들어온 사람들은 {total}몇 명{ques_trailing}',
            equation=gen.EqnRef('eqn02_3', order1_k, n_people_btw_k),

            env=gen.fnmap(
                place=place,
                person=person,
                rank_name=rank_name,
                line=line,
                sent1=sent1,
                sent2_1=sent2_1,
                sent2_2=sent2_2,
                sent3=sent3,
                sent5=sent5,
                sent6=sent6,
                order1=order1_k,
                n_people_btw=n_people_btw_k,
                name1=name1_k,
                name2=name2_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_3_7(selector, tokenpool, clskey):
    ''' 학생들이 달리기 시합을 하였다. 
    유정이가 가장 마지막에 들어왔고, 은정이는 5번째로 들어왔다. 
    은정이와 유정이 사이에 8명이 들어왔을 때, 달리기를 한 사람은 모두 몇 명인가? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    line = random.choice(['', '줄을', '한 줄로 줄을', '일렬로', '한 줄로'])
    sent1 = random.choice(['을 때, ', '있다면, '])
    sent2_choice = ['는', '가']
    sent3 = random.choice(['', '들'])
    sent5 = random.choice(['이 ', '의 사람들이 '])
    sent6 = random.choice(['맨 ', '가장 '])

    # new variables
    person = selector.get(clskey.person)
    rank_name = selector.get(clskey.rank)
    name1 = selector.get(clskey.name) # 유정
    name2 = selector.get(clskey.name) # 은정
    place = selector.get(clskey.place)

    max_idx = 100
    order1 = random.randint(1, max_idx-2) # 5번째
    order2 = random.randint(order1+2, max_idx)
    n_people_btw = random.randint(1, order2-order1-1) # 8명
    random.shuffle(sent2_choice)
    sent2_1 = sent2_choice[0]
    sent2_2 = sent2_choice[1]

    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    order1_k = tokenpool.new(order1)
    n_people_btw_k = tokenpool.new(n_people_btw)

    return gen.build(
            body='학생들이 달리기 시합을 하였{sent_trailing} '\
            '{name1}{#이?}가 {sent6}마지막에 들어왔고, {name2}{#이?}는 '\
            f'{random.choice([order1, gen.korutil.num2korord(order1)])}번째로 들어왔다.',
            question='{name1}{#이?}와 {name2}{#이?} 사이에 {n_people_btw}명{sent5}들어왔을 때, '\
            '달리기를 한 사람은 {total}몇 명{ques_trailing}',
            equation=gen.EqnRef('eqn02_3', order1_k, n_people_btw_k),

            env=gen.fnmap(
                place=place,
                person=person,
                rank_name=rank_name,
                line=line,
                sent1=sent1,
                sent2_1=sent2_1,
                sent2_2=sent2_2,
                sent3=sent3,
                sent5=sent5,
                sent6=sent6,
                order1=order1_k,
                n_people_btw=n_people_btw_k,
                name1=name1_k,
                name2=name2_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_3_8(selector, tokenpool, clskey):
    ''' 유정이와 은정이가 계단에 서 있습니다. 
    유정이가 가장 높은 계단에 서 있고, 은정이는 아래서 5번째에 서 있습니다. 
    은정이와 유정이 사이에 8명이 서 있을 때, 계단에 있는 사람은 모두 몇 명인가? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    line = random.choice(['', '줄을', '한 줄로 줄을', '일렬로', '한 줄로'])
    sent1 = random.choice(['을 때, ', '있다면, '])
    sent2_choice = ['는', '가']
    sent3 = random.choice(['', '들'])
    sent5 = random.choice(['이 ', '의 사람들이 '])
    sent6 = random.choice(['맨 ', '가장 '])

    # new variables
    person = selector.get(clskey.person)
    rank_name = selector.get(clskey.rank)
    name1 = selector.get(clskey.name) # 유정
    name2 = selector.get(clskey.name) # 은정
    place = selector.get(clskey.place)

    max_idx = 100
    order1 = random.randint(1, max_idx-2) # 5번째
    order2 = random.randint(order1+2, max_idx)
    n_people_btw = random.randint(1, order2-order1-1) # 8명
    random.shuffle(sent2_choice)
    sent2_1 = sent2_choice[0]
    sent2_2 = sent2_choice[1]

    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    order1_k = tokenpool.new(order1)
    n_people_btw_k = tokenpool.new(n_people_btw)

    return gen.build(
            body='{name1}{#이?}와 {name2}{#이?}가 계단에 서 있습니다. '\
            '{name1}{#이?}가 가장 높은 계단에 서 있고, {name2}{#이?}는 아래서 '\
            f'{random.choice([order1, gen.korutil.num2korord(order1)])}번째에 서 있습니다.',
            question='{name1}{#이?}와 {name2}{#이?} 사이에 {n_people_btw}명{sent5}서 있을 때, '\
            '계단에 있는 사람은 {total}몇 명{ques_trailing}',
            equation=gen.EqnRef('eqn02_3', order1_k, n_people_btw_k),

            env=gen.fnmap(
                place=place,
                person=person,
                rank_name=rank_name,
                line=line,
                sent1=sent1,
                sent2_1=sent2_1,
                sent2_2=sent2_2,
                sent3=sent3,
                sent5=sent5,
                sent6=sent6,
                order1=order1_k,
                n_people_btw=n_people_btw_k,
                name1=name1_k,
                name2=name2_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_3_9(selector, tokenpool, clskey):
    ''' 유정이와 은정이는 같은 아파트(건물)에 살고 있습니다. 
    유정이는 가장 윗층에 살고 있고, 은정이는 5층에 살고 있습니다. 
    은정이와 유정이 사이에 8개층이 있다면, 이 아파트(건물)은 모두 몇 층입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    line = random.choice(['', '줄을', '한 줄로 줄을', '일렬로', '한 줄로'])
    sent1 = random.choice(['을 때, ', '있다면, '])
    sent2_choice = ['는', '가']
    sent3 = random.choice(['', '들'])
    sent5 = random.choice(['이 ', '의 사람들이 '])
    sent6 = random.choice(['맨 ', '가장 '])
    sent7 = random.choice(['아파트', '건물', '빌딩'])

    # new variables
    person = selector.get(clskey.person)
    rank_name = selector.get(clskey.rank)
    name1 = selector.get(clskey.name) # 유정
    name2 = selector.get(clskey.name) # 은정
    place = selector.get(clskey.place)

    max_idx = 100
    order1 = random.randint(1, max_idx-2) # 5번째
    order2 = random.randint(order1+2, max_idx)
    n_people_btw = random.randint(1, order2-order1-1) # 8명
    random.shuffle(sent2_choice)
    sent2_1 = sent2_choice[0]
    sent2_2 = sent2_choice[1]

    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    order1_k = tokenpool.new(order1)
    n_people_btw_k = tokenpool.new(n_people_btw)

    return gen.build(
            body='{name1}{#이?}와 {name2}{#이?}는 같은 {sent7}에 살고 있습니다. '\
            '{name1}{#이?}는 {sent6}윗층에 살고, {name2}{#이?}는 {order1}층에 살고 있습니다.',
            question='{name1}{#이?}와 {name2}{#이?} 사이에 {n_people_btw}개층이 있다면, '\
            '이 {sent7}{#은} {total}몇 층{ques_trailing}',
            equation=gen.EqnRef('eqn02_3', order1_k, n_people_btw_k),

            env=gen.fnmap(
                place=place,
                person=person,
                rank_name=rank_name,
                line=line,
                sent1=sent1,
                sent2_1=sent2_1,
                sent2_2=sent2_2,
                sent3=sent3,
                sent5=sent5,
                sent6=sent6,
                sent7=sent7,
                order1=order1_k,
                n_people_btw=n_people_btw_k,
                name1=name1_k,
                name2=name2_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_3_10(selector, tokenpool, clskey):
    ''' 유정이와 은정이가 달리기를 하고 있습니다. 
    유정이가 가장 느리고, 은정이는 앞에서부터 5번째로 달리고 있습니다. 
    둘 사이에 다른 8명의 학생들이 있을 때, 달리기를 하고 있는 학생은 모두 몇 명입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    line = random.choice(['', '줄을', '한 줄로 줄을', '일렬로', '한 줄로'])
    sent1 = random.choice(['을 때, ', '있다면, '])
    sent2_choice = ['는', '가']
    sent3 = random.choice(['', '들'])
    sent5 = random.choice(['이 ', '의 사람들이 '])
    sent6 = random.choice(['맨 ', '가장 '])

    # new variables
    person = selector.get(clskey.person)
    rank_name = selector.get(clskey.rank)
    name1 = selector.get(clskey.name) # 유정
    name2 = selector.get(clskey.name) # 은정
    place = selector.get(clskey.place)

    max_idx = 100
    order1 = random.randint(1, max_idx-2) # 5번째
    order2 = random.randint(order1+2, max_idx)
    n_people_btw = random.randint(1, order2-order1-1) # 8명
    random.shuffle(sent2_choice)
    sent2_1 = sent2_choice[0]
    sent2_2 = sent2_choice[1]

    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    order1_k = tokenpool.new(order1)
    n_people_btw_k = tokenpool.new(n_people_btw)

    return gen.build(
            body='{name1}{#이?}와 {name2}{#이?}가 달리기를 하고 있습니다. '\
            '{name1}{#이?}가 가장 느리고, {name2}{#이?}는 앞에서부터 '\
            f'{random.choice([order1, gen.korutil.num2korord(order1)])}번째로 달리고 있습니다.',
            question='둘 사이에 다른 {n_people_btw}명의 학생들이 있다면, '\
            '달리기를 하고 있는 학생은 {total}몇 명{ques_trailing}',
            equation=gen.EqnRef('eqn02_3', order1_k, n_people_btw_k),

            env=gen.fnmap(
                place=place,
                person=person,
                rank_name=rank_name,
                line=line,
                sent1=sent1,
                sent2_1=sent2_1,
                sent2_2=sent2_2,
                sent3=sent3,
                sent5=sent5,
                sent6=sent6,
                order1=order1_k,
                n_people_btw=n_people_btw_k,
                name1=name1_k,
                name2=name2_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))


######################################################################################
################################## problem 04 ########################################

# @gen.problems.register
def prob02_4_1(selector, tokenpool, clskey):
    ''' 윤기는 왼쪽에서 7번째 열, 오른쪽에서 13번째 열, 앞에서 8번째 줄, 뒤에서
    14번째 줄에 서서 체조를 하고 있습니다. 각 줄마다 서 있는 학생의 수가
    같다고 할 때, 체조를 하고 있는 학생은 모두 몇 명입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exercise = random.choice(['운동', '체조'])

    # new variables
    person = selector.get(clskey.person)
    name1 = selector.get(clskey.name)

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    front_back = ['앞', '뒤']
    random.shuffle(front_back)
    front_back_0 = front_back[0]
    front_back_1 = front_back[1]

    idx_left = random.randint(1, 100)
    idx_right = random.randint(1, 100)
    idx_front = random.randint(1, 100)
    idx_down = random.randint(1, 100)

    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_front)
    idx_front_k = tokenpool.new(idx_front)
    idx_down_k = tokenpool.new(idx_down)

    return gen.build(
            body=' '.join([
                '{name1}{#이?}는 {left_right_0}에서 {idx_left}번째 열, '\
                '{left_right_1}에서 {idx_right}번째 열, {front_back_0}에서 '\
                '{idx_front}번째 줄, {front_back_1}에서 {idx_down}번째 줄에 '\
                '서서 {exercise}{#를} 하고 있습니다.'
            ]),
            question='각 줄마다 서 있는 {person}의 수가 같다고 할 때, '\
            '{exercise}{#를} 하고 있는 {person}은 {total}몇 명 {ques_trailing}',
            equation=gen.EqnRef('eqn02_4', idx_left_k, idx_right_k, idx_front_k, idx_down_k),

            env=gen.fnmap(
                exercise=exercise,
                person=person,
                name1=name1,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                front_back_0=front_back_0,
                front_back_1=front_back_1,
                idx_left=idx_left_k,
                idx_right=idx_right_k,
                idx_front=idx_front_k,
                idx_down=idx_down_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_4_2(selector, tokenpool, clskey):
    ''' 사람들이 모여있을 때, 윤기는 왼쪽에서 7번째 열, 오른쪽에서 13번째 열, 앞에서 8번째 줄, 뒤에서
    14번째 줄에 서 있습니다. 각 줄마다 서 있는 학생의 수가
    같다고 할 때, 모여있는 사람들의 수를 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    exercise = random.choice(['운동', '체육', '체조'])

    # new variables
    person = selector.get(clskey.person)
    name1 = selector.get(clskey.name)

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    front_back = ['앞', '뒤']
    random.shuffle(front_back)
    front_back_0 = front_back[0]
    front_back_1 = front_back[1]

    idx_left = random.randint(1, 100)
    idx_right = random.randint(1, 100)
    idx_front = random.randint(1, 100)
    idx_down = random.randint(1, 100)

    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_front)
    idx_front_k = tokenpool.new(idx_front)
    idx_down_k = tokenpool.new(idx_down)

    return gen.build(
            body=' '.join([
                '사람들이 모여있을 때, {name1}{#이?}는 {left_right_0}에서 {idx_left}번째 열, '\
                '{left_right_1}에서 {idx_right}번째 열, {front_back_0}에서 '\
                '{idx_front}번째 줄, {front_back_1}에서 {idx_down}번째 줄에 서 있습니다.'
            ]),
            question='각 줄마다 서 있는 {person}의 수가 같다고 할 때, '\
            '모여있는 {person}의 수를 {ques_trailing}',
            equation=gen.EqnRef('eqn02_4', idx_left_k, idx_right_k, idx_front_k, idx_down_k),

            env=gen.fnmap(
                exercise=exercise,
                person=person,
                name1=name1,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                front_back_0=front_back_0,
                front_back_1=front_back_1,
                idx_left=idx_left_k,
                idx_right=idx_right_k,
                idx_front=idx_front_k,
                idx_down=idx_down_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

def prob02_4_3(selector, tokenpool, clskey):
    ''' 윤기를 포함하여 왼쪽에는 7명, 오른쪽에는 13명, 앞에는 8명, 
    뒤에는 14명의 사람이 있습니다. 
    모든 줄에 서있는 사람들의 수가 같다고 할 때, 서있는 사람들의 수는 모두 몇 명입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['를 구하시오.', '를 구하세요.', '는 모두 몇 명입니까?'])
    exercise = random.choice(['운동', '체육', '체조'])
    sent1 = random.choice(['', '들'])

    # new variables
    person = selector.get(clskey.person)
    name1 = selector.get(clskey.name)

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    front_back = ['앞', '뒤']
    random.shuffle(front_back)
    front_back_0 = front_back[0]
    front_back_1 = front_back[1]

    idx_left = random.randint(1, 100)
    idx_right = random.randint(1, 100)
    idx_front = random.randint(1, 100)
    idx_down = random.randint(1, 100)

    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_front)
    idx_front_k = tokenpool.new(idx_front)
    idx_down_k = tokenpool.new(idx_down)

    return gen.build(
            body=' '.join([
                '{name1}{#를} 포함하여 {left_right_0}에는 {idx_left}명, '\
                '{left_right_1}에는 {idx_right}명, {front_back_0}에는 '\
                '{idx_front}명, {front_back_1}에는 {idx_down}명의 {person}{sent1}이 있습니다.'
            ]),
            question='모든 줄에 서 있는 {person}{sent1}의 수가 같다고 할 때, '\
            '서 있는 {person}{sent1}의 수{ques_trailing}',
            equation=gen.EqnRef('eqn02_4', idx_left_k, idx_right_k, idx_front_k, idx_down_k),

            env=gen.fnmap(
                exercise=exercise,
                person=person,
                name1=name1,
                sent1=sent1,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                front_back_0=front_back_0,
                front_back_1=front_back_1,
                idx_left=idx_left_k,
                idx_right=idx_right_k,
                idx_front=idx_front_k,
                idx_down=idx_down_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_4_4(selector, tokenpool, clskey):
    ''' 윤기의 좌석이 왼쪽에서 ~~ 있습니다. 
    각 줄의 좌석 수가 같다면, 좌석의 수는 총 몇 개인가? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exercise = random.choice(['운동', '체육', '체조'])
    seat = random.choice(['자리', '좌석', '자동차'])
    sent1 = random.choice(['', '들'])


    # new variables
    person = selector.get(clskey.person)
    name1 = selector.get(clskey.name)

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    front_back = ['앞', '뒤']
    random.shuffle(front_back)
    front_back_0 = front_back[0]
    front_back_1 = front_back[1]

    idx_left = random.randint(1, 100)
    idx_right = random.randint(1, 100)
    idx_front = random.randint(1, 100)
    idx_down = random.randint(1, 100)

    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_front)
    idx_front_k = tokenpool.new(idx_front)
    idx_down_k = tokenpool.new(idx_down)

    return gen.build(
            body=' '.join([
                '{name1}{#이?}{#의} {seat}{#이} {left_right_0}에서 {idx_left}번째 ',\
                '{left_right_1}에서 {idx_right}번째, {front_back_0}에서 {idx_front}번째, '\
                '{front_back_1}에서 {idx_down}번째에 있습니다.'
            ]),
            question='각 줄의 {seat} 수가 같다면, {seat}의 수는 {total}몇 개{ques_trailing}',
            equation=gen.EqnRef('eqn02_4', idx_left_k, idx_right_k, idx_front_k, idx_down_k),

            env=gen.fnmap(
                exercise=exercise,
                person=person,
                name1=name1,
                seat=seat,
                sent1=sent1,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                front_back_0=front_back_0,
                front_back_1=front_back_1,
                idx_left=idx_left_k,
                idx_right=idx_right_k,
                idx_front=idx_front_k,
                idx_down=idx_down_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_4_5(selector, tokenpool, clskey):
    ''' 책꽂이에 책이 왼쪽에서 ~~ 에 꽂혀져 있다. 
    각 줄에 같은 수의 책이 꽂혀 있다고 할 때, 책꽂이에 꽂혀있는 책은 모두 몇 권입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exercise = random.choice(['운동', '체육', '체조'])
    seat = random.choice(['자리', '좌석', '자동차'])
    sent1 = random.choice(['', '들'])


    # new variables
    shelf = selector.get(clskey.shelf)
    book = selector.get(clskey.book)

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    front_back = ['앞', '뒤']
    random.shuffle(front_back)
    front_back_0 = front_back[0]
    front_back_1 = front_back[1]

    idx_left = random.randint(1, 100)
    idx_right = random.randint(1, 100)
    idx_front = random.randint(1, 100)
    idx_down = random.randint(1, 100)

    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_front)
    idx_front_k = tokenpool.new(idx_front)
    idx_down_k = tokenpool.new(idx_down)

    return gen.build(
            body=' '.join([
                '{shelf}에 {book}{#이} {left_right_0}에서 {idx_left}번째 '\
                '{left_right_1}에서 {idx_right}번째, {front_back_0}에서 {idx_front}번째, '\
                '{front_back_1}에서 {idx_down}번째에 있습니다.'
            ]),
            question='각 줄에 같은 수의 {book}{#이} 꽂혀 있다고 할 때, '\
            '모든 {book}의 수는 몇 개{ques_trailing}',
            equation=gen.EqnRef('eqn02_4', idx_left_k, idx_right_k, idx_front_k, idx_down_k),

            env=gen.fnmap(
                shelf=shelf,
                book=book,
                seat=seat,
                sent1=sent1,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                front_back_0=front_back_0,
                front_back_1=front_back_1,
                idx_left=idx_left_k,
                idx_right=idx_right_k,
                idx_front=idx_front_k,
                idx_down=idx_down_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_4_6(selector, tokenpool, clskey):
    ''' 모든 줄의 인원 수를 똑같이 맞추어 행진을 하고 있습니다. 
    윤기가 속한 위치가 왼쪽에서 ~~ 일 때, 행진을 하고 있는 사람은 모두 몇 명입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exercise = random.choice(['운동', '체육', '체조'])
    seat = random.choice(['자리', '좌석', '자동차'])
    sent1 = random.choice(['', '들'])


    # new variables
    name1 = selector.get(clskey.name)

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    front_back = ['앞', '뒤']
    random.shuffle(front_back)
    front_back_0 = front_back[0]
    front_back_1 = front_back[1]

    idx_left = random.randint(1, 100)
    idx_right = random.randint(1, 100)
    idx_front = random.randint(1, 100)
    idx_down = random.randint(1, 100)

    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_front)
    idx_front_k = tokenpool.new(idx_front)
    idx_down_k = tokenpool.new(idx_down)

    return gen.build(
            body='모든 줄의 인원 수를 똑같이 맞추어 행진을 하고 있습니다.',
            question='{name1}{#가} 속한 위치가 {left_right_0}에서 {idx_left}번째 '\
                '{left_right_1}에서 {idx_right}번째, {front_back_0}에서 {idx_front}번째, '\
                '{front_back_1}에서 {idx_down}번째일 때, 행진을 하고 있는 사람은 {total}몇 명{ques_trailing}',
            equation=gen.EqnRef('eqn02_4', idx_left_k, idx_right_k, idx_front_k, idx_down_k),

            env=gen.fnmap(
                name1=name1,
                sent1=sent1,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                front_back_0=front_back_0,
                front_back_1=front_back_1,
                idx_left=idx_left_k,
                idx_right=idx_right_k,
                idx_front=idx_front_k,
                idx_down=idx_down_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_4_7(selector, tokenpool, clskey):
    ''' 윤기가 직사각형 모양의 바둑판에 바둑알을 놓았다. 
    바둑알의 위치가 왼쪽에서 ~~일 때, 바둑판의 모든 칸의 개수를 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    exercise = random.choice(['운동', '체육', '체조'])
    seat = random.choice(['자리', '좌석', '자동차'])
    sent1 = random.choice(['', '들'])


    # new variables
    name1 = selector.get(clskey.name)

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    front_back = ['앞', '뒤']
    random.shuffle(front_back)
    front_back_0 = front_back[0]
    front_back_1 = front_back[1]

    idx_left = random.randint(1, 100)
    idx_right = random.randint(1, 100)
    idx_front = random.randint(1, 100)
    idx_down = random.randint(1, 100)

    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_front)
    idx_front_k = tokenpool.new(idx_front)
    idx_down_k = tokenpool.new(idx_down)

    return gen.build(
            body='{name1}{#이?}가 직사각형 모양의 바둑판에 바둑알을 놓았{sent_trailing}',
            question='바둑알의 위치가 {left_right_0}에서 {idx_left}번째 '\
                '{left_right_1}에서 {idx_right}번째, {front_back_0}에서 {idx_front}번째, '\
                '{front_back_1}에서 {idx_down}번째일 때, '\
                '바둑판의 모든 칸의 개수를 {ques_trailing}',
            equation=gen.EqnRef('eqn02_4', idx_left_k, idx_right_k, idx_front_k, idx_down_k),

            env=gen.fnmap(
                name1=name1,
                sent1=sent1,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                front_back_0=front_back_0,
                front_back_1=front_back_1,
                idx_left=idx_left_k,
                idx_right=idx_right_k,
                idx_front=idx_front_k,
                idx_down=idx_down_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_4_8(selector, tokenpool, clskey):
    ''' 매장에 인형이 전시되어 있다. 
    어느 인형이 왼쪽에서 ~~에 전시되어 있을 때, 
    매장에 전시되어 있는 인형의 개수는 모두 몇 개인가? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exercise = random.choice(['운동', '체육', '체조'])
    seat = random.choice(['자리', '좌석', '자동차'])
    sent1 = random.choice(['', '들'])


    # new variables
    name1 = selector.get(clskey.name)

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    front_back = ['앞', '뒤']
    random.shuffle(front_back)
    front_back_0 = front_back[0]
    front_back_1 = front_back[1]

    idx_left = random.randint(1, 100)
    idx_right = random.randint(1, 100)
    idx_front = random.randint(1, 100)
    idx_down = random.randint(1, 100)

    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_front)
    idx_front_k = tokenpool.new(idx_front)
    idx_down_k = tokenpool.new(idx_down)

    return gen.build(
            body='매장에 인형이 전시되어 있다. ',
            question='어느 인형이 {left_right_0}에서 {idx_left}번째 '\
            '{left_right_1}에서 {idx_right}번째, {front_back_0}에서 {idx_front}번째, '\
            '{front_back_1}에서 {idx_down}번째일 때, '\
            '매장에 전시되어 있는 인형의 개수는 {total}몇 개 {ques_trailing}',
            equation=gen.EqnRef('eqn02_4', idx_left_k, idx_right_k, idx_front_k, idx_down_k),

            env=gen.fnmap(
                name1=name1,
                sent1=sent1,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                front_back_0=front_back_0,
                front_back_1=front_back_1,
                idx_left=idx_left_k,
                idx_right=idx_right_k,
                idx_front=idx_front_k,
                idx_down=idx_down_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_4_9(selector, tokenpool, clskey):
    ''' 모든 줄의 인원을 똑같이 맞추어 체조를 하고 있다. 
    윤기는 왼쪽에서 ~~ 에 서서 체조를 하고 있다면, 
    체조를 하고 있는 학생은 모두 몇 명입니까?? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exercise = random.choice(['운동', '체육', '체조'])
    seat = random.choice(['자리', '좌석', '자동차'])
    sent1 = random.choice(['', '들'])


    # new variables
    name1 = selector.get(clskey.name)

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    front_back = ['앞', '뒤']
    random.shuffle(front_back)
    front_back_0 = front_back[0]
    front_back_1 = front_back[1]

    idx_left = random.randint(1, 100)
    idx_right = random.randint(1, 100)
    idx_front = random.randint(1, 100)
    idx_down = random.randint(1, 100)

    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_front)
    idx_front_k = tokenpool.new(idx_front)
    idx_down_k = tokenpool.new(idx_down)

    return gen.build(
            body='모든 줄의 인원을 똑같이 맞추어 {exercise}{#를} 하고 있다.',
            question='{name1}{#이?}는 {left_right_0}에서 {idx_left}번째 '\
            '{left_right_1}에서 {idx_right}번째, {front_back_0}에서 {idx_front}번째, '\
            '{front_back_1}에서 {idx_down}번째에 서서 체조를 하고 있다면, '\
            '체조를 하고 있는 학생은 {total}몇 명{ques_trailing}',
            equation=gen.EqnRef('eqn02_4', idx_left_k, idx_right_k, idx_front_k, idx_down_k),

            env=gen.fnmap(
                name1=name1,
                sent1=sent1,
                exercise=exercise,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                front_back_0=front_back_0,
                front_back_1=front_back_1,
                idx_left=idx_left_k,
                idx_right=idx_right_k,
                idx_front=idx_front_k,
                idx_down=idx_down_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_4_10(selector, tokenpool, clskey):
    ''' 각 줄마다 서 있는 학생의 수를 맞추어 체조를 하고 있다. 
    윤기는 왼쪽에서 ~~에 서서 체조를 하고 있다면, 
    체조를 하고 있는 학생은 모두 몇 명입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exercise = random.choice(['운동', '체육', '체조'])
    seat = random.choice(['자리', '좌석', '자동차'])
    sent1 = random.choice(['', '들'])


    # new variables
    name1 = selector.get(clskey.name)

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    front_back = ['앞', '뒤']
    random.shuffle(front_back)
    front_back_0 = front_back[0]
    front_back_1 = front_back[1]

    idx_left = random.randint(1, 100)
    idx_right = random.randint(1, 100)
    idx_front = random.randint(1, 100)
    idx_down = random.randint(1, 100)

    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_front)
    idx_front_k = tokenpool.new(idx_front)
    idx_down_k = tokenpool.new(idx_down)

    return gen.build(
            body='각 줄마다 서 있는 학생의 수를 맞추어 {exercise}{#를} 하고 있다.',
            question='{name1}{#이?}는 {left_right_0}에서 {idx_left}번째 '\
            '{left_right_1}에서 {idx_right}번째, {front_back_0}에서 {idx_front}번째, '\
            '{front_back_1}에서 {idx_down}번째에 서서 체조를 하고 있다면, '\
            '체조를 하고 있는 학생은 {total}몇 명{ques_trailing}',
            equation=gen.EqnRef('eqn02_4', idx_left_k, idx_right_k, idx_front_k, idx_down_k),

            env=gen.fnmap(
                name1=name1,
                sent1=sent1,
                exercise=exercise,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                front_back_0=front_back_0,
                front_back_1=front_back_1,
                idx_left=idx_left_k,
                idx_right=idx_right_k,
                idx_front=idx_front_k,
                idx_down=idx_down_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))


######################################################################################
################################## problem 05 ########################################

# @gen.problems.register
def prob02_5_1(selector, tokenpool, clskey):
    ''' 도서관에 똑같은 책장이 28개 있습니다. 각 책장은 6층이고, 각 층마다 꽂혀있는
    책의 수는 같습니다. 영어책은 어느 책장의 한 층의 왼쪽에서 9번째, 오른쪽에서 11번째에 
    꽂혀 있습니다.
    도서관의 책장에 꽂혀 있는 책은 모두 몇 권입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    same = random.choice([' ', '같은 ', '똑같은 ', '똑 같은 '])

    # new variables
    shelf = selector.get(clskey.shelf)
    floor = selector.get(clskey.floor)
    bookstore = selector.get(clskey.bookstore)
    book = selector.get(clskey.book)

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    n_shelves = random.randint(1, 100)
    n_floors = random.randint(1, 100)
    idx_left = random.randint(1, 100)
    idx_right = random.randint(1, 100)
    idx_up = random.randint(1, 100)
    idx_down = random.randint(1, 100)

    n_floors_k = tokenpool.new(n_floors)
    n_shelves_k = tokenpool.new(n_shelves)
    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_right)
    idx_up_k = tokenpool.new(idx_up)
    idx_down_k = tokenpool.new(idx_down)

    return gen.build(
            body=' '.join([
                '{bookstore}에 {same}{shelf}{#가} {n_shelves}개 있습니다. '\
                '각 {shelf}{#는} {n_floors}{floor}이고, 각 {floor}마다 꽂혀 있는 '\
                '{book}의 수는 같습니다. {book}{#은} 어느 {shelf}의 한 {floor}의 '\
                '{left_right_0}에서 {idx_left}번째, {left_right_1}에서 {idx_right}번째에 '\
                '꽂혀 있습니다.'
            ]),
            question='{bookstore}의 {shelf}에 꽂혀 있는 {book}{#은} {total}몇 권{ques_trailing}',
            equation=gen.EqnRef('eqn02_5', idx_left_k, idx_right_k, n_floors_k, n_shelves_k),

            env=gen.fnmap(
                same=same,
                bookstore=bookstore,
                shelf=shelf,
                floor=floor,
                book=book,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                n_shelves=n_shelves_k,
                n_floors=n_floors_k,
                idx_left=idx_left_k,
                idx_right=idx_right_k,
                idx_up=idx_up_k,
                idx_down=idx_down_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_5_2(selector, tokenpool, clskey):
    ''' 높이가 6층인 책장 28개 중 어느 책장의 한 층에 영어책이 왼쪽에서 ~~에 꽂혀 있습니다. 
    모든 층에 꽂혀있는 책의 수가 같을 때, 이 책장에 꽂혀있는 책의 개수는 모두 몇 권입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    same = random.choice([' ', '같은 ', '똑같은 ', '똑 같은 '])

    # new variables
    shelf = selector.get(clskey.shelf) # 책장
    floor = selector.get(clskey.floor) # 층
    bookstore = selector.get(clskey.bookstore) # 도서관
    book = selector.get(clskey.book) # 영어책

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    n_shelves = random.randint(1, 100) # 28개
    n_floors = random.randint(1, 100) # 6층
    idx_left = random.randint(1, 100) # 9번째
    idx_right = random.randint(1, 100) # 11번째

    n_floors_k = tokenpool.new(n_floors)
    n_shelves_k = tokenpool.new(n_shelves)
    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_right)

    return gen.build(
            body=' '.join([
                '높이가 {n_floors}{floor}인 {shelf} {n_shelves}개 중 어느 {shelf}의 한 층에 {book}{#이} '\
                '{left_right_0}에서 {idx_left}번째, {left_right_1}에서 {idx_right}번째에 꽂혀 있{sent_trailing}'
            ]),
            question='{floor}마다 꽂혀 있는 {book}의 수는 같을 때, 이 {shelf}에 꽂혀있는 {book}의 개수는 {total}몇 권{ques_trailing}',
            equation=gen.EqnRef('eqn02_5', idx_left_k, idx_right_k, n_floors_k, n_shelves_k),

            env=gen.fnmap(
                same=same,
                bookstore=bookstore,
                shelf=shelf,
                floor=floor,
                book=book,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                n_shelves=n_shelves_k,
                n_floors=n_floors_k,
                idx_left=idx_left_k,
                idx_right=idx_right_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_5_3(selector, tokenpool, clskey):
    ''' 6층짜리 책장 28개가 있고, 각 층마다 꽂혀있는 책의 수가 같습니다. 왼쪽에서 ~~에 영어책이 꽂혀있다면, 
    이 책장에 꽂혀있는 모든 책의 개수를 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    same = random.choice([' ', '같은 ', '똑같은 ', '똑 같은 '])

    # new variables
    shelf = selector.get(clskey.shelf) # 책장
    floor = selector.get(clskey.floor) # 층
    bookstore = selector.get(clskey.bookstore) # 도서관
    book = selector.get(clskey.book) # 영어책

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    n_shelves = random.randint(1, 100) # 28개
    n_floors = random.randint(1, 100) # 6층
    idx_left = random.randint(1, 100) # 9번째
    idx_right = random.randint(1, 100) # 11번째

    n_floors_k = tokenpool.new(n_floors)
    n_shelves_k = tokenpool.new(n_shelves)
    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_right)

    return gen.build(
            body=' '.join([
                '{n_floors}{floor}짜리 {shelf} {n_shelves}개가 있고, 각 {floor}마다 꽂혀있는 책의 개수가 같{sent_trailing} '
            ]),
            question='{left_right_0}에서 {idx_left}번째, {left_right_1}에서 {idx_right}번째에 {book}{#이} 꽂혀 있다면, '\
            '이 {shelf}에 꽂혀있는 {total}책의 개수를 {ques_trailing}',
            equation=gen.EqnRef('eqn02_5', idx_left_k, idx_right_k, n_floors_k, n_shelves_k),

            env=gen.fnmap(
                same=same,
                bookstore=bookstore,
                shelf=shelf,
                floor=floor,
                book=book,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                n_shelves=n_shelves_k,
                n_floors=n_floors_k,
                idx_left=idx_left_k,
                idx_right=idx_right_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_5_4(selector, tokenpool, clskey):
    ''' 도서관에 똑같은 책장이 28개 있고, 각 책장은 6층입니다. 각 층마다 꽂혀있는 책의 수가 같을 때, 
    영어책이 어느 책장의 한 층에 왼쪽에서 ~~에 있다면 모든 책장에 꽂혀있는 책의 개수를 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    same = random.choice([' ', '같은 ', '똑같은 ', '똑 같은 '])

    # new variables
    shelf = selector.get(clskey.shelf) # 책장
    floor = selector.get(clskey.floor) # 층
    bookstore = selector.get(clskey.bookstore) # 도서관
    book = selector.get(clskey.book) # 영어책

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    n_shelves = random.randint(1, 100) # 28개
    n_floors = random.randint(1, 100) # 6층
    idx_left = random.randint(1, 100) # 9번째
    idx_right = random.randint(1, 100) # 11번째

    n_floors_k = tokenpool.new(n_floors)
    n_shelves_k = tokenpool.new(n_shelves)
    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_right)

    return gen.build(
            body='{bookstore}에 {same}{shelf}{#가} {n_shelves}개가 있고, 각 {shelf}{#은} {n_floors}{floor}입니다.',
            question='각 {floor}마다 꽂혀있는 책의 수가 같을 때, {book}{#이} 어느 {shelf}의 한 {floor}에 '\
            '{left_right_0}에서 {idx_left}번째, {left_right_1}에서 {idx_right}번째에 {book}{#이} 있다면, '\
            '모든 {shelf}에 꽂혀있는 책의 개수를 {ques_trailing}',
            equation=gen.EqnRef('eqn02_5', idx_left_k, idx_right_k, n_floors_k, n_shelves_k),

            env=gen.fnmap(
                same=same,
                bookstore=bookstore,
                shelf=shelf,
                floor=floor,
                book=book,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                n_shelves=n_shelves_k,
                n_floors=n_floors_k,
                idx_left=idx_left_k,
                idx_right=idx_right_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_5_5(selector, tokenpool, clskey):
    ''' 한 층에 영어책이 왼쪽에서 ~~에 꽂혀있는 책장이 있습니다. 
    모든 층에 같은 수의 책이 있다면, 6층짜리 28개의 책장에는 모두 몇 권의 책이 꽂혀있습니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['습니까?', '는지 구하시오.', '는지 구하세요.', '는지 구해 보세요.', '는가?'])
    same = random.choice([' ', '같은 ', '똑같은 ', '똑 같은 '])

    # new variables
    shelf = selector.get(clskey.shelf) # 책장
    floor = selector.get(clskey.floor) # 층
    bookstore = selector.get(clskey.bookstore) # 도서관
    book = selector.get(clskey.book) # 영어책

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    n_shelves = random.randint(1, 100) # 28개
    n_floors = random.randint(1, 100) # 6층
    idx_left = random.randint(1, 100) # 9번째
    idx_right = random.randint(1, 100) # 11번째

    n_floors_k = tokenpool.new(n_floors)
    n_shelves_k = tokenpool.new(n_shelves)
    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_right)

    return gen.build(
            body='한 {floor}에 {book}{#이} {left_right_0}에서 {idx_left}번째, {left_right_1}에서 '\
            '{idx_right}번째에 꽂혀있는 {shelf}{#이} 있{sent_trailing}',
            question='각 {floor}마다 꽂혀있는 책의 수가 같다면, {n_floors}{floor}짜리 {n_shelves}개의 '\
            '{shelf}에는 모두 몇 권의 책이 꽂혀있{ques_trailing}',
            equation=gen.EqnRef('eqn02_5', idx_left_k, idx_right_k, n_floors_k, n_shelves_k),

            env=gen.fnmap(
                same=same,
                bookstore=bookstore,
                shelf=shelf,
                floor=floor,
                book=book,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                n_shelves=n_shelves_k,
                n_floors=n_floors_k,
                idx_left=idx_left_k,
                idx_right=idx_right_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_5_6(selector, tokenpool, clskey):
    ''' 6층으로 된 책장의 한 층에 영어책이 왼쪽에서 ~~에 꽂혀있습니다. 
    모양이 똑같은 책장이 28개 있고 모든 층에 꽂혀있는 책의 수가 같을 때, 모든 책장에 꽂혀있는 책의 개수를 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    same = random.choice([' ', '같은 ', '똑같은 ', '똑 같은 '])

    # new variables
    shelf = selector.get(clskey.shelf) # 책장
    floor = selector.get(clskey.floor) # 층
    bookstore = selector.get(clskey.bookstore) # 도서관
    book = selector.get(clskey.book) # 영어책

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    n_shelves = random.randint(1, 100) # 28개
    n_floors = random.randint(1, 100) # 6층
    idx_left = random.randint(1, 100) # 9번째
    idx_right = random.randint(1, 100) # 11번째

    n_floors_k = tokenpool.new(n_floors)
    n_shelves_k = tokenpool.new(n_shelves)
    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_right)

    return gen.build(
            body='{n_floors}{floor}으로 된 {shelf}의 한 {floor}에 {book}{#이} '\
            '{left_right_0}에서 {idx_left}번째, {left_right_1}에서 {idx_right}번째에 꽂혀있{sent_trailing}',
            question='모양이 똑같은 {shelf}{#이} {n_shelves}개 있고, 모든 {floor}에 꽂혀있는 '\
            '{book}의 수가 같을 때, 모든 {shelf}에 있는 책의 개수를 {ques_trailing}',
            equation=gen.EqnRef('eqn02_5', idx_left_k, idx_right_k, n_floors_k, n_shelves_k),

            env=gen.fnmap(
                same=same,
                bookstore=bookstore,
                shelf=shelf,
                floor=floor,
                book=book,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                n_shelves=n_shelves_k,
                n_floors=n_floors_k,
                idx_left=idx_left_k,
                idx_right=idx_right_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))


# @gen.problems.register
def prob02_5_7(selector, tokenpool, clskey):
    ''' 어느 한 책장에 영어책이 왼쪽에서 ~~에 꽂혀 있습니다. 
    이 책장은 6층이고 총 28개의 똑같은 책장이 있습니다. 
    책장의 각 층에 꽂혀있는 책의 수가 같다면, 모든 책장에 꽂혀있는 책은 모두 몇 권입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    same = random.choice([' ', '같은 ', '똑같은 ', '똑 같은 '])

    # new variables
    shelf = selector.get(clskey.shelf) # 책장
    floor = selector.get(clskey.floor) # 층
    bookstore = selector.get(clskey.bookstore) # 도서관
    book = selector.get(clskey.book) # 영어책

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    n_shelves = random.randint(1, 100) # 28개
    n_floors = random.randint(1, 100) # 6층
    idx_left = random.randint(1, 100) # 9번째
    idx_right = random.randint(1, 100) # 11번째

    n_floors_k = tokenpool.new(n_floors)
    n_shelves_k = tokenpool.new(n_shelves)
    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_right)

    return gen.build(
            body='어느 한 {shelf}에 {book}{#이} {left_right_0}에서 {idx_left}번째, '\
            '{left_right_1}에서 {idx_right}번째에 꽂혀 있{sent_trailing} '\
            '이 {shelf}{#은} {n_floors}{floor}이고 총 {n_shelves}개의 똑같은 {shelf}{#이} 있습니다.',
            question='{shelf}의 각 {floor}에 꽂혀있는 {book}의 수가 같다면, 모든 {shelf}에 꽂혀있는 '\
            '{book}{#은} 모두 몇 권{ques_trailing}',
            equation=gen.EqnRef('eqn02_5', idx_left_k, idx_right_k, n_floors_k, n_shelves_k),

            env=gen.fnmap(
                same=same,
                bookstore=bookstore,
                shelf=shelf,
                floor=floor,
                book=book,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                n_shelves=n_shelves_k,
                n_floors=n_floors_k,
                idx_left=idx_left_k,
                idx_right=idx_right_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))


# @gen.problems.register
def prob02_5_8(selector, tokenpool, clskey):
    ''' 각 층마다 꽂혀있는 책의 수가 같은 책장이 있습니다. 
    이 책장에 영어책이 왼쪽에서 ~~에 꽂혀있고, 책장의 높이는 6층입니다. 
    똑같은 책장이 28개가 있다고 한다면, 모든 책장에 있는 책의 개수는 모두 몇 권입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    same = random.choice([' ', '같은 ', '똑같은 ', '똑 같은 '])

    # new variables
    shelf = selector.get(clskey.shelf) # 책장
    floor = selector.get(clskey.floor) # 층
    bookstore = selector.get(clskey.bookstore) # 도서관
    book = selector.get(clskey.book) # 영어책

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    n_shelves = random.randint(1, 100) # 28개
    n_floors = random.randint(1, 100) # 6층
    idx_left = random.randint(1, 100) # 9번째
    idx_right = random.randint(1, 100) # 11번째

    n_floors_k = tokenpool.new(n_floors)
    n_shelves_k = tokenpool.new(n_shelves)
    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_right)

    return gen.build(
            body='각 {floor}마다 꽂혀있는 {book}의 수가 같은 {shelf}{#이} 있습니다. '\
            '이 {shelf}에 {book}{#이} {left_right_0}에서 {idx_left}번째, '\
            '{left_right_1}에서 {idx_right}번째에 꽂혀 있고, {shelf}의 높이는 {n_floors}{floor}입니다.',
            question='똑같은 {shelf}{#이} {n_shelves}개가 있다고 한다면, '\
            '모든 {shelf}에 있는 {book}의 개수는 모두 몇 권{ques_trailing}',
            equation=gen.EqnRef('eqn02_5', idx_left_k, idx_right_k, n_floors_k, n_shelves_k),

            env=gen.fnmap(
                same=same,
                bookstore=bookstore,
                shelf=shelf,
                floor=floor,
                book=book,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                n_shelves=n_shelves_k,
                n_floors=n_floors_k,
                idx_left=idx_left_k,
                idx_right=idx_right_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_5_9(selector, tokenpool, clskey):
    ''' 도서관에 6층 책장이 28개 있습니다. 
    책장에는 영어책이 왼쪽에서 ~~에 꽂혀 있습니다. 
    각 층마다 꽂혀있는 책의 수가 같다고 한다면, 도서관에 꽂혀있는 책의 수는 모두 몇 권입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    same = random.choice([' ', '같은 ', '똑같은 ', '똑 같은 '])

    # new variables
    shelf = selector.get(clskey.shelf) # 책장
    floor = selector.get(clskey.floor) # 층
    bookstore = selector.get(clskey.bookstore) # 도서관
    book = selector.get(clskey.book) # 영어책

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    n_shelves = random.randint(1, 100) # 28개
    n_floors = random.randint(1, 100) # 6층
    idx_left = random.randint(1, 100) # 9번째
    idx_right = random.randint(1, 100) # 11번째

    n_floors_k = tokenpool.new(n_floors)
    n_shelves_k = tokenpool.new(n_shelves)
    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_right)

    return gen.build(
            body='{bookstore}에 {n_floors}{floor} {shelf}{#이} {n_shelves}개 있습니다. '\
            '{shelf}에는 {book}{#이} {left_right_0}에서 {idx_left}번째, '\
            '{left_right_1}에서 {idx_right}번째에 꽂혀 있습니다.',
            question='각 {floor}마다 꽂혀있는 {book}의 수가 같다고 한다면, '\
            '{bookstore}에 꽂혀있는 {book}의 수는 모두 몇 권{ques_trailing}',
            equation=gen.EqnRef('eqn02_5', idx_left_k, idx_right_k, n_floors_k, n_shelves_k),

            env=gen.fnmap(
                same=same,
                bookstore=bookstore,
                shelf=shelf,
                floor=floor,
                book=book,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                n_shelves=n_shelves_k,
                n_floors=n_floors_k,
                idx_left=idx_left_k,
                idx_right=idx_right_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob02_5_10(selector, tokenpool, clskey):
    ''' 영어책이 왼쪽에서 ~~에 꽂혀있는 책장이 있습니다. 
    이 책장은 6층이고, 모두 28개입니다. 
    각 층마다 꽂혀있는 책의 수가 같다면, 모든 책장에 꽂혀있는 책의 개수를 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    same = random.choice([' ', '같은 ', '똑같은 ', '똑 같은 '])

    # new variables
    shelf = selector.get(clskey.shelf) # 책장
    floor = selector.get(clskey.floor) # 층
    bookstore = selector.get(clskey.bookstore) # 도서관
    book = selector.get(clskey.book) # 영어책

    left_right = ['왼쪽', '오른쪽']
    random.shuffle(left_right)
    left_right_0 = left_right[0]
    left_right_1 = left_right[1]

    n_shelves = random.randint(1, 100) # 28개
    n_floors = random.randint(1, 100) # 6층
    idx_left = random.randint(1, 100) # 9번째
    idx_right = random.randint(1, 100) # 11번째

    n_floors_k = tokenpool.new(n_floors)
    n_shelves_k = tokenpool.new(n_shelves)
    idx_left_k = tokenpool.new(idx_left)
    idx_right_k = tokenpool.new(idx_right)

    return gen.build(
            body='{book}{#이} {left_right_0}에서 {idx_left}번째, '\
            '{left_right_1}에서 {idx_right}번째에 꽂혀있는 {shelf}{#이} 있습니다. '\
            '이 {shelf}{#은} {n_floors}{floor}이고, 모두 {n_shelves}개입니다.',
            question='각 {floor}마다 꽂혀있는 {book}의 수가 같다고 한다면, '\
            '모든 {shelf}에 꽂혀있는 {book}의 개수를 {ques_trailing}',
            equation=gen.EqnRef('eqn02_5', idx_left_k, idx_right_k, n_floors_k, n_shelves_k),

            env=gen.fnmap(
                same=same,
                bookstore=bookstore,
                shelf=shelf,
                floor=floor,
                book=book,
                left_right_0=left_right_0,
                left_right_1=left_right_1,
                n_shelves=n_shelves_k,
                n_floors=n_floors_k,
                idx_left=idx_left_k,
                idx_right=idx_right_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

######################################################################################
################################## problem 06_2 ######################################

# @gen.problems.register
def prob06_2_1(selector, tokenpool, clskey):
    ''' 어떤 수에서 46을 빼야하는데 잘못하여 59를 뺐더니 43이 되었습니다.
    바르게 계산한 결과를 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    exp_sub_add = ['빼야 하는데', '빼야 할 것을', '빼야 될 것을', '빼야 되는데', '더해야 하는데', '더해야 할 것을', '더할 것을']
    exp_wrong = random.choice(['잘못하여', '잘못해서', '잘못해'])

    # new variables
    orig_num = random.randint(1, 100)
    wrong_num = random.randint(1, 100)
    wrong_result = random.randint(1, 100)

    case1 = random.choice(exp_sub_add)
    case2 = random.choice(['뺐더니', '더했더니'])

    orig_num_k = tokenpool.new(orig_num) # 46
    wrong_num_k = tokenpool.new(wrong_num) # 59
    wrong_result_k = tokenpool.new(wrong_result) # 43
    case1_k = tokenpool.new(case1)
    case2_k = tokenpool.new(case2)

    if '빼' in case1:
        if case2 == '뺐더니':
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, -orig_num_k)
        else:
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, -orig_num_k)
    else:
        if case2 == '뺐더니':
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, orig_num_k)
        else:
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, orig_num_k)

    return gen.build(
            body=' '.join(['어떤 수에서 {orig_num}을 {case1} {exp_wrong} {wrong_num}를 '\
            '{case2} {wrong_result}이 되었습니다.'
            ]),
            question='바르게 계산한 결과를 {ques_trailing}',
            equation=equation_temp,

            env=gen.fnmap(
                exp_sub_add=exp_sub_add,
                exp_wrong=exp_wrong,
                orig_num=orig_num_k,
                wrong_num=wrong_num_k,
                wrong_result=wrong_result_k,
                case1=case1_k,
                case2=case2_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob06_2_2(selector, tokenpool, clskey):
    ''' 어떤 수보다 59만큼 작은 수는 43입니다.
    어떤 수보다 46만큼 작은 수를 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    exp_sub_add = ['빼야 하는데', '빼야 할 것을', '빼야 될 것을', '빼야 되는데', '더해야 하는데', '더해야 할 것을', '더할 것을']
    exp_wrong = random.choice(['잘못하여', '잘못해서', '잘못해', '실수로'])

    # new variables
    orig_num = random.randint(1, 100)
    wrong_num = random.randint(1, 100)
    wrong_result = random.randint(1, 100)

    case1 = random.choice(exp_sub_add)
    case2 = random.choice(['뺐더니', '더했더니'])

    orig_num_k = tokenpool.new(orig_num) # 46
    wrong_num_k = tokenpool.new(wrong_num) # 59
    wrong_result_k = tokenpool.new(wrong_result) # 43
    case1_k = tokenpool.new(case1)
    case2_k = tokenpool.new(case2)

    if '빼' in case1:
        big_or_small_1 = '작은'
        if case2 == '뺐더니':
            big_or_small_0 = '작은'
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, -orig_num_k)
        else:
            big_or_small_0 = '큰'
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, -orig_num_k)
    else:
        big_or_small_1 = '작은'
        if case2 == '뺐더니':
            big_or_small_0 = '작은'
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, orig_num_k)
        else:
            big_or_small_0 = '큰'
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, orig_num_k)

    return gen.build(
            body='어떤 수보다 {wrong_num}만큼 {big_or_small_0} 수는 {wrong_result}입니다.',
            question='어떤 수보다 {orig_num}만큼 {big_or_small_1} 수를 {ques_trailing}',
            equation=equation_temp,

            env=gen.fnmap(
                exp_sub_add=exp_sub_add,
                exp_wrong=exp_wrong,
                orig_num=orig_num_k,
                wrong_num=wrong_num_k,
                wrong_result=wrong_result_k,
                case1=case1_k,
                case2=case2_k,
                big_or_small_0=big_or_small_0,
                big_or_small_1=big_or_small_1,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob06_2_3(selector, tokenpool, clskey):
    ''' 어떤 수에서(와) 59를 빼면 43입니다.
    어떤 수에서 46을 빼면 얼마인지 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exp_sub_add = ['빼면', '더하면']

    # new variables
    orig_num = random.randint(1, 100)
    wrong_num = random.randint(1, 100)
    wrong_result = random.randint(1, 100)

    case1 = random.choice(exp_sub_add)
    case2 = random.choice(['뺐더니', '더했더니'])

    orig_num_k = tokenpool.new(orig_num) # 46
    wrong_num_k = tokenpool.new(wrong_num) # 59
    wrong_result_k = tokenpool.new(wrong_result) # 43
    case1_k = tokenpool.new(case1)
    case2_k = tokenpool.new(case2)

    if '빼' in case1:
        big_or_small_1 = '작은'
        if case2 == '뺐더니':
            big_or_small_0 = '작은'
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, -orig_num_k)
        else:
            big_or_small_0 = '큰'
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, -orig_num_k)
    else:
        big_or_small_1 = '작은'
        if case2 == '뺐더니':
            big_or_small_0 = '작은'
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, orig_num_k)
        else:
            big_or_small_0 = '큰'
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, orig_num_k)

    return gen.build(
            body='어떤 수에서 {wrong_num}{#를} {case2} {wrong_result}입니다.',
            question='어떤 수에서 {orig_num}{#를} {case1} 얼마{ques_trailing}',
            equation=equation_temp,

            env=gen.fnmap(
                exp_sub_add=exp_sub_add,
                orig_num=orig_num_k,
                wrong_num=wrong_num_k,
                wrong_result=wrong_result_k,
                case1=case1_k,
                case2=case2_k,
                big_or_small_0=big_or_small_0,
                big_or_small_1=big_or_small_1,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob06_2_4(selector, tokenpool, clskey):
    ''' 59를 어떤 수에서 빼면 43이 나올 때, 46을 어떤 수에서 빼면 얼마인지 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exp_sub_add = ['빼면', '더하면']

    # new variables
    orig_num = random.randint(1, 100)
    wrong_num = random.randint(1, 100)
    wrong_result = random.randint(1, 100)

    case1 = random.choice(exp_sub_add)
    case2 = random.choice(['뺐더니', '더했더니'])

    orig_num_k = tokenpool.new(orig_num) # 46
    wrong_num_k = tokenpool.new(wrong_num) # 59
    wrong_result_k = tokenpool.new(wrong_result) # 43
    case1_k = tokenpool.new(case1)
    case2_k = tokenpool.new(case2)

    if '빼' in case1:
        big_or_small_1 = '작은'
        if case2 == '뺐더니':
            big_or_small_0 = '작은'
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, -orig_num_k)
        else:
            big_or_small_0 = '큰'
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, -orig_num_k)
    else:
        big_or_small_1 = '작은'
        if case2 == '뺐더니':
            big_or_small_0 = '작은'
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, orig_num_k)
        else:
            big_or_small_0 = '큰'
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, orig_num_k)

    return gen.build(
            body='{wrong_num}{#를} 어떤 수에서 {case2} {wrong_result}이 나온다면, ',
            question='{orig_num}{#를} 어떤 수에서 {case1} 얼마{ques_trailing}',
            equation=equation_temp,

            env=gen.fnmap(
                exp_sub_add=exp_sub_add,
                orig_num=orig_num_k,
                wrong_num=wrong_num_k,
                wrong_result=wrong_result_k,
                case1=case1_k,
                case2=case2_k,
                big_or_small_0=big_or_small_0,
                big_or_small_1=big_or_small_1,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))


# @gen.problems.register
def prob06_2_5(selector, tokenpool, clskey):
    ''' 어떤 수는 43에서 59를 더한 수입니다. 
    어떤 수에서 46을 빼면 얼마인지 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exp_sub_add = ['빼면', '더하면']

    # new variables
    orig_num = random.randint(1, 100)
    wrong_num = random.randint(1, 100)
    wrong_result = random.randint(1, 100)

    case1 = random.choice(exp_sub_add)
    case2 = random.choice(['뺐더니', '더했더니'])

    orig_num_k = tokenpool.new(orig_num) # 46
    wrong_num_k = tokenpool.new(wrong_num) # 59
    wrong_result_k = tokenpool.new(wrong_result) # 43
    case1_k = tokenpool.new(case1)
    case2_k = tokenpool.new(case2)

    if '빼' in case1:
        if case2 == '뺐더니':
            case2_reverse = '더한'
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, -orig_num_k)
        else:
            case2_reverse = '뺀'
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, -orig_num_k)
    else:
        if case2 == '뺐더니':
            case2_reverse = '더한'
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, orig_num_k)
        else:
            case2_reverse = '뺀'
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, orig_num_k)

    case2_reverse_k=tokenpool.new(case2_reverse)

    return gen.build(
            body='어떤 수는 {wrong_result}에서 {wrong_num}{#을} {case2_reverse} 수입니다.',
            question='어떤 수에서 {orig_num}{#를} {case1} 얼마{ques_trailing}',
            equation=equation_temp,

            env=gen.fnmap(
                exp_sub_add=exp_sub_add,
                orig_num=orig_num_k,
                wrong_num=wrong_num_k,
                wrong_result=wrong_result_k,
                case1=case1_k,
                case2=case2_k,
                case2_reverse=case2_reverse_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))


# @gen.problems.register
def prob06_2_6(selector, tokenpool, clskey):
    ''' 어떤 수 빼기 59는 43과 같습니다. 
    어떤 수에서 46을 빼면 얼마인지 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exp_sub_add = ['빼면', '더하면']

    # new variables
    orig_num = random.randint(1, 100)
    wrong_num = random.randint(1, 100)
    wrong_result = random.randint(1, 100)

    case1 = random.choice(exp_sub_add)
    case2 = random.choice(['뺐더니', '더했더니'])

    orig_num_k = tokenpool.new(orig_num) # 46
    wrong_num_k = tokenpool.new(wrong_num) # 59
    wrong_result_k = tokenpool.new(wrong_result) # 43
    case1_k = tokenpool.new(case1)
    case2_k = tokenpool.new(case2)

    if '빼' in case1:
        if case2 == '뺐더니':
            sent1 = '빼기'
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, -orig_num_k)
        else:
            sent1 = '더하기'
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, -orig_num_k)
    else:
        if case2 == '뺐더니':
            sent1 = '빼기'
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, orig_num_k)
        else:
            sent1 = '더하기'
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, orig_num_k)

    sent1_k=tokenpool.new(sent1)

    return gen.build(
            body='어떤 수 {sent1} {wrong_num}{#는} {wrong_result}{#과} 같습니다.',
            question='어떤 수에서 {orig_num}{#를} {case1} 얼마{ques_trailing}',
            equation=equation_temp,

            env=gen.fnmap(
                exp_sub_add=exp_sub_add,
                orig_num=orig_num_k,
                wrong_num=wrong_num_k,
                wrong_result=wrong_result_k,
                case1=case1_k,
                case2=case2_k,
                sent1=sent1_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob06_2_7(selector, tokenpool, clskey):
    ''' 어떤 수에 59를 빼서 43을 만들었습니다. 
    어떤 수에서 46을 빼면 얼마인지 구해 보세요. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exp_sub_add = ['빼면', '더하면']

    # new variables
    orig_num = random.randint(1, 100)
    wrong_num = random.randint(1, 100)
    wrong_result = random.randint(1, 100)

    case1 = random.choice(exp_sub_add)
    case2 = random.choice(['뺐더니', '더했더니'])

    orig_num_k = tokenpool.new(orig_num) # 46
    wrong_num_k = tokenpool.new(wrong_num) # 59
    wrong_result_k = tokenpool.new(wrong_result) # 43
    case1_k = tokenpool.new(case1)
    case2_k = tokenpool.new(case2)

    if '빼' in case1:
        if case2 == '뺐더니':
            sent1 = '빼서'
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, -orig_num_k)
        else:
            sent1 = '더해서'
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, -orig_num_k)
    else:
        if case2 == '뺐더니':
            sent1 = '빼서'
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, orig_num_k)
        else:
            sent1 = '더해서'
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, orig_num_k)

    sent1_k=tokenpool.new(sent1)

    return gen.build(
            body='어떤 수에 {wrong_num}{#를} {sent1} {wrong_result}{#를} 만들었습니다.',
            question='어떤 수에서 {orig_num}{#를} {case1} 얼마{ques_trailing}',
            equation=equation_temp,

            env=gen.fnmap(
                exp_sub_add=exp_sub_add,
                orig_num=orig_num_k,
                wrong_num=wrong_num_k,
                wrong_result=wrong_result_k,
                case1=case1_k,
                case2=case2_k,
                sent1=sent1_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob06_2_8(selector, tokenpool, clskey):
    ''' 상자에서 사과를 59개 꺼내면 43개가 남습니다. 
    만약 46개를 빼면 상자에 몇 개의 사과가 남는지 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    exp_sub_add = ['꺼내면', '넣으면']

    # new variables
    container = selector.get(clskey.container)
    item = selector.get(clskey.fruit)

    orig_num = random.randint(1, 100)
    wrong_num = random.randint(1, 100)
    wrong_result = random.randint(1, 100)

    case1 = random.choice(exp_sub_add)
    case2 = random.choice(exp_sub_add)

    orig_num_k = tokenpool.new(orig_num) # 46
    wrong_num_k = tokenpool.new(wrong_num) # 59
    wrong_result_k = tokenpool.new(wrong_result) # 43
    case1_k = tokenpool.new(case1)
    case2_k = tokenpool.new(case2)

    if '꺼내면' in case1:
        sent3 = '남는지'
        if case2 == '꺼내면':
            sent2 = '남습니다. '
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, -orig_num_k)
        else:
            sent2 = '됩니다. '
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, -orig_num_k)
    else:
        sent3 = '되는지'
        if case2 == '꺼내면':
            sent2 = '남습니다. '
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, orig_num_k)
        else:
            sent2 = '됩니다. '
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, orig_num_k)

    sent2_k=tokenpool.new(sent2)
    sent3_k=tokenpool.new(sent3)

    return gen.build(
            body='{container}에서 {item} {wrong_num}개를 {case2} {wrong_result}개가 {sent2}',
            question='만약 {orig_num}개를 {case1} {container}에 몇 개의 {item}{#이} {sent3} {ques_trailing}',
            equation=equation_temp,

            env=gen.fnmap(
                exp_sub_add=exp_sub_add,
                orig_num=orig_num_k,
                wrong_num=wrong_num_k,
                wrong_result=wrong_result_k,
                container=container,
                item=item,
                case1=case1_k,
                case2=case2_k,
                sent2=sent2_k,
                sent3=sent3_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob06_2_9(selector, tokenpool, clskey):
    ''' 정국이가 친구들에게 59권의 책을 나눠주면 43개가 남습니다. 
    만약 46개의 책을 나눠주면 몇 권의 책이 남는지 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    exp_sub_add = ['나눠주면', '받으면']

    # new variables
    name1 = selector.get(clskey.name)
    n = random.randint(0, 1)
    if n == 0:
        item = selector.get(clskey.fruit)
    elif n == 1:
        item = selector.get(clskey.book)
    unit = item.of("unit")

    orig_num = random.randint(1, 100)
    wrong_num = random.randint(1, 100)
    wrong_result = random.randint(1, 100)

    case1 = random.choice(exp_sub_add)
    case2 = random.choice(exp_sub_add)

    orig_num_k = tokenpool.new(orig_num) # 46
    wrong_num_k = tokenpool.new(wrong_num) # 59
    wrong_result_k = tokenpool.new(wrong_result) # 43
    case1_k = tokenpool.new(case1)
    case2_k = tokenpool.new(case2)
    item_k = tokenpool.new(case2)
    item_k.unit = unit

    if '나눠주면' in case1:
        sent3 = '남는지'
        if case2 == '나눠주면':
            sent2 = '남습니다. '
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, -orig_num_k)
        else:
            sent2 = '됩니다. '
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, -orig_num_k)
    else:
        sent3 = '되는지'
        if case2 == '나눠주면':
            sent2 = '남습니다. '
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, orig_num_k)
        else:
            sent2 = '됩니다. '
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, orig_num_k)

    sent2_k=tokenpool.new(sent2)
    sent3_k=tokenpool.new(sent3)

    return gen.build(
            body='{name1}{#이?}가 친구들에게 {item} {wrong_num}{unit}{#을} {case2} {wrong_result}{unit}가 {sent2}',
            question='만약 {orig_num}{unit}의 {item}{#을} {case1} 몇 {unit}의 {item}{#이} {sent3} {ques_trailing}',
            equation=equation_temp,

            env=gen.fnmap(
                exp_sub_add=exp_sub_add,
                orig_num=orig_num_k,
                wrong_num=wrong_num_k,
                wrong_result=wrong_result_k,
                item=item,
                name1=name1,
                case1=case1_k,
                case2=case2_k,
                sent3=sent3_k,
                sent2=sent2_k,
                unit=unit,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob06_2_10(selector, tokenpool, clskey):
    ''' 정국이는 민영이에게 책을 59권 받고, 태형이에게 43권 받았습니다. 
    만약, 정국이가 지민이에게 46권의 책을 준다면 남는 책은 몇 권입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exp_sub_add_0 = ['받고', '주고']
    exp_sub_add_1 = ['준다면 ', '받는다면 ']

    # new variables
    name1 = selector.get(clskey.name)
    name2 = selector.get(clskey.name)
    name3 = selector.get(clskey.name)
    name4 = selector.get(clskey.name)
    n = random.randint(0, 1)
    if n == 0:
        item = selector.get(clskey.fruit)
    elif n == 1:
        item = selector.get(clskey.book)
    unit = item.of("unit")

    orig_num = random.randint(1, 100)
    wrong_num = random.randint(1, 100)
    wrong_result = random.randint(1, 100)

    case1 = random.choice(exp_sub_add_0)
    case2 = random.choice(exp_sub_add_1)

    orig_num_k = tokenpool.new(orig_num) # 46
    wrong_num_k = tokenpool.new(wrong_num) # 59
    wrong_result_k = tokenpool.new(wrong_result) # 43
    case1_k = tokenpool.new(case1)
    case2_k = tokenpool.new(case2)
    item_k = tokenpool.new(case2)
    item_k.unit = unit

    if '준다면' in case2:
        if '받고' in case1:
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, -orig_num_k)
        else:
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, -orig_num_k)
    else:
        if '받고' in case1:
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, orig_num_k)
        else:
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, orig_num_k)

    return gen.build(
            body='{name1}{#이?}는 {name2}에게 {item}{#을} {wrong_num}{unit}{#을} {case1}, '\
            '{name3}{#이?}에게 {wrong_result}{unit}{#을} 받았습니다.',
            question='만약 {name1}{#이?}가 {name4}{#이?}에게 {orig_num}{unit}의 {item}{#을} '\
            '{case2}{name1}{#이?}가 가지고 있는 {item}{#은} {total}몇 {unit}{ques_trailing}',
            equation=equation_temp,

            env=gen.fnmap(
                orig_num=orig_num_k,
                wrong_num=wrong_num_k,
                wrong_result=wrong_result_k,
                item=item,
                name1=name1,
                name2=name2,
                name3=name3,
                name4=name4,
                case1=case1_k,
                case2=case2_k,
                unit=unit,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob06_2_11(selector, tokenpool, clskey):
    ''' 59와 43을 더한 값에서 46을 뺀(빼면) 결과값은 얼마인가? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    exp_sub_add = ['더한 ', '뺀 ']
    sent1 = random.choice(['값', '결과값'])

    # new variables
    orig_num = random.randint(1, 100)
    wrong_num = random.randint(1, 100)
    wrong_result = random.randint(1, 100)

    case1 = random.choice(exp_sub_add)
    case2 = random.choice(exp_sub_add)

    orig_num_k = tokenpool.new(orig_num) # 46
    wrong_num_k = tokenpool.new(wrong_num) # 59
    wrong_result_k = tokenpool.new(wrong_result) # 43
    case1_k = tokenpool.new(case1)
    case2_k = tokenpool.new(case2)

    if '뺀' in case2:
        if '더한' in case1:
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, -orig_num_k)
        else:
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, -orig_num_k)
    else:
        if '더한' in case1:
            equation_temp = gen.EqnRef('sum', wrong_result_k, wrong_num_k, orig_num_k)
        else:
            equation_temp = gen.EqnRef('sum', wrong_result_k, -wrong_num_k, orig_num_k)

    return gen.build(
            body='',
            question='{wrong_result}에서 {wrong_num}{#을} {case1}값에서 '\
            '{orig_num}{#을} {case2}{sent1}은 얼마{ques_trailing}',
            equation=equation_temp,

            env=gen.fnmap(
                orig_num=orig_num_k,
                wrong_num=wrong_num_k,
                wrong_result=wrong_result_k,
                case1=case1_k,
                case2=case2_k,
                sent1=sent1,

                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))





######################################################################################
################################## problem 07_2 ######################################

# @gen.problems.register
def prob07_2_1(selector, tokenpool, clskey):
    ''' 네 수 A, B, C, D가 있습니다. A는 27입니다. B는 A보다 7 큰 수입니다. C는 B보다
    9 작은 수입니다. D는 C의 2배인 수입니다.
    가장 큰 수는 어느 것입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])

    # new variables
    final_sent = ['입니다.', '이고,', ',']
    size = ['큰', '작은']
    divide = ['인', ' 작은']
    variables = ['A', 'B', 'C', 'D']

    big_or_small_0 = random.choice(size)
    big_or_small_1 = random.choice(size)
    divide = random.choice(divide)

    num1 = random.randint(1, 100)
    num2 = random.randint(1, 100)
    num3 = random.randint(1, 100)
    num4 = random.randint(1, 10)

    big_or_small_0_k = tokenpool.new(big_or_small_0)
    big_or_small_1_k = tokenpool.new(big_or_small_1)
    num1_k = tokenpool.new(num1)
    num2_k = tokenpool.new(num2)
    num3_k = tokenpool.new(num3)
    num4_k = tokenpool.new(num4)

    if big_or_small_0 == '큰':
        B = num1+num2
        if big_or_small_1_k == '큰':
            C = num2+num3
            if divide == '인':
                D = int(C*num4)
                equation_temp = gen.EqnRef('eqn07_2', num1_k, num1_k, num2_k, num2_k, num3_k, num2_k, num3_k, num4_k, num1_k, num1_k, num2_k, num2_k, num3_k, num2_k, num3_k, num4_k, *variables)
            else:
                D = C//num4
                equation_temp = gen.EqnRef('eqn07_2', num1_k, num1_k, num2_k, num2_k, num3_k, num2_k, num3_k, num4_k, num1_k, num1_k, num2_k, num2_k, num3_k, num2_k, num3_k, num4_k, *variables)
        else:
            C = num2-num3
            if divide == '인':
                D = int(C*num4)
                equation_temp = gen.EqnRef('eqn07_2', num1_k, num1_k, num2_k, num2_k, -num3_k, num2_k, -num3_k, num4_k, num1_k, num1_k, num2_k, num2_k, -num3_k, num2_k, -num3_k, num4_k, *variables)
            else:
                D = C//num4
                equation_temp = gen.EqnRef('eqn07_2', num1_k, num1_k, num2_k, num2_k, -num3_k, num2_k, -num3_k, num4_k, num1_k, num1_k, num2_k, num2_k, -num3_k, num2_k, -num3_k, num4_k, *variables)
    else:
        B = num1-num2
        if big_or_small_1_k == '큰':
            C = num2num3
            if divide == '인':
                D = int(C*num4)
                equation_temp = gen.EqnRef('eqn07_2', num1_k, num1_k, -num2_k, num2_k, num3_k, num2_k, num3_k, num4_k, num1_k, num1_k, -num2_k, num2_k, num3_k, num2_k, num3_k, num4_k, *variables)
            else:
                D = C//num4
                equation_temp = gen.EqnRef('eqn07_2', num1_k, num1_k, -num2_k, num2_k, num3_k, num2_k, num3_k, num4_k, num1_k, num1_k, -num2_k, num2_k, num3_k, num2_k, num3_k, num4_k, *variables)
        else:
            C = num2-num3
            if divide == '인':
                D = int(C*num4)
                equation_temp = gen.EqnRef('eqn07_2', num1_k, num1_k, -num2_k, num2_k, -num3_k, num2_k, -num3_k, num4_k, num1_k, num1_k, -num2_k, num2_k, -num3_k, num2_k, -num3_k, num4_k, *variables)
            else:
                D = C//num4
                equation_temp = gen.EqnRef('eqn07_2', num1_k, num1_k, -num2_k, num2_k, -num3_k, num2_k, -num3_k, num4_k, num1_k, num1_k, -num2_k, num2_k, -num3_k, num2_k, -num3_k, num4_k, *variables)


    return gen.build(
            body=' '.join([f'네 수 A, B, C, D가 있습니다. {variables[0]}는 '+\
            f'{num1_k}{random.choice(final_sent)}{variables[1]}는 {variables[0]}보다 '+\
            f'{num2_k} {big_or_small_0} 수{random.choice(final_sent)} {variables[2]}는 '+\
            f'{variables[1]}보다 {num3_k} {big_or_small_1} 수{random.choice(final_sent)} '+\
            f'{variables[3]}는 {variables[2]}의 {num4_k}배{divide} 수입니다.'
            ]),
            question='가장 큰 수는 어느 것입니까?',
            equation=equation_temp,

            env=gen.fnmap(
                final_sent=final_sent,
                size=size,
                divide=divide,
                variables=variables,
                big_or_small_0=big_or_small_0_k,
                big_or_small_1=big_or_small_1_k,
                num1=num1_k,
                num2=num2_k,
                num3=num3_k,
                num4=num4_k,

                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))


######################################################################################
################################## problem 08_2 ######################################

# @gen.problems.register
def prob08_2_1(selector, tokenpool, clskey):
    ''' 한 변의 길이가 5cm인 정삼각형의 둘레는 몇 cm입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent_exp1 = random.choice(['의 길이가', '이,'])

    # new variables
    length = random.randint(1, 100)
    n = random.randint(1, 6)

    length_k = tokenpool.new(length)
    n_k = tokenpool.new(n)

    return gen.build(
            body='',
            question='한 변{sent_exp1} {length}cm인 '\
            f'정{gen.korutil.num2kor(n+3)}각형의 '\
            '둘레는 몇 cm{ques_trailing}',
            equation=gen.EqnRef('eqn08_2', length_k, n_k),

            env=gen.fnmap(
                sent_exp1=sent_exp1,
                length=length_k,
                n=n_k,

                ques_trailing=ques_trailing
            ))


# @gen.problems.register
def prob08_2_2(selector, tokenpool, clskey):
    '''정삼각형의 한 변의 길이가 5cm이다. 둘레의 길이는 몇 cm입니까?'''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent_exp1 = random.choice(['의 길이가 ', '이,', '의 길이는 '])
    sent_exp2 = random.choice(['는 ', '의 길이는 '])
    sent_exp3 = random.choice(['이다.', '입니다.'])

    # new variables
    length = random.randint(1, 100)
    n = random.randint(1, 6)

    length_k = tokenpool.new(length)
    n_k = tokenpool.new(n)

    return gen.build(
            body=f'정{gen.korutil.num2kor(n+3)}각형의 '\
            '한 변{sent_exp1} {length}cm{sent_exp3}',
            question='둘레는 몇 cm{ques_trailing}',
            equation=gen.EqnRef('eqn08_2', length_k, n_k),

            env=gen.fnmap(
                sent_exp1=sent_exp1,
                sent_exp3=sent_exp3,
                length=length_k,
                n=n_k,

                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob08_2_3(selector, tokenpool, clskey):
    ''' 정삼각형의 한 변의 길이가 5cm일 때, 둘레의 길이는 몇 cm입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent_exp1 = random.choice(['의 ', ' '])
    sent_exp2 = random.choice(['는 ', '의 길이는 '])
    sent_exp3 = random.choice(['한 ', ''])

    # new variables
    length = random.randint(1, 100)
    n = random.randint(1, 6)

    length_k = tokenpool.new(length)
    n_k = tokenpool.new(n)

    return gen.build(
            body='',
            question=f'정{gen.korutil.num2kor(n+3)}각형'\
            '{sent_exp1}{sent_exp3}변의 길이가 {length}cm일 때, '\
            '둘레{sent_exp2}몇 cm{ques_trailing}',
            equation=gen.EqnRef('eqn08_2', length_k, n_k),

            env=gen.fnmap(
                sent_exp1=sent_exp1,
                sent_exp2=sent_exp2,
                sent_exp3=sent_exp3,
                length=length_k,
                n=n_k,

                ques_trailing=ques_trailing
            ))


# @gen.problems.register
def prob08_2_4(selector, tokenpool, clskey):
    '''한 변의 길이가 5cm인 정삼각형이 있다. 둘레의 길이는 몇 cm입니까?'''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent_exp2 = random.choice(['는 ', '의 길이는 '])
    sent_exp3 = random.choice(['한 ', ''])

    # new variables
    length = random.randint(1, 100)
    n = random.randint(1, 6)

    length_k = tokenpool.new(length)
    n_k = tokenpool.new(n)

    return gen.build(
            body='{sent_exp3}변의 길이가 {length}cm인 '
            f'정{gen.korutil.num2kor(n+3)}각형이 '\
            '있{sent_trailing}',
            question='둘레{sent_exp2}몇 cm{ques_trailing}',
            equation=gen.EqnRef('eqn08_2', length_k, n_k),

            env=gen.fnmap(
                sent_exp2=sent_exp2,
                sent_exp3=sent_exp3,
                length=length_k,
                n=n_k,

                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob08_2_5(selector, tokenpool, clskey):
    '''한 변의 길이가 5cm인 정삼각형의 둘레를 구하시오.'''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent_exp1 = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    sent_exp2 = random.choice(['를 ', '의 길이를 '])
    sent_exp3 = random.choice(['한 ', ''])

    # new variables
    length = random.randint(1, 100)
    n = random.randint(1, 6)

    length_k = tokenpool.new(length)
    n_k = tokenpool.new(n)

    return gen.build(
            body='',
            question='{sent_exp3}변의 길이가 {length}cm인 '\
            f'정{gen.korutil.num2kor(n+3)}각형의 '\
            '둘레{sent_exp2} {sent_exp1}',
            equation=gen.EqnRef('eqn08_2', length_k, n_k),

            env=gen.fnmap(
                sent_exp1=sent_exp1,
                sent_exp2=sent_exp2,
                sent_exp3=sent_exp3,
                length=length_k,
                n=n_k,

                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob08_2_6(selector, tokenpool, clskey):
    '''한 변의 길이가 5cm인 정삼각형의 둘레의 길이는 얼마입니까?'''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent_exp2 = random.choice(['는 ', '의 길이는 '])
    sent_exp3 = random.choice(['한 ', ''])

    # new variables
    length = random.randint(1, 100)
    n = random.randint(1, 6)

    length_k = tokenpool.new(length)
    n_k = tokenpool.new(n)

    return gen.build(
            body='',
            question='{sent_exp3}변의 길이가 {length}cm인 '\
            f'정{gen.korutil.num2kor(n+3)}각형의 '\
            '둘레{sent_exp2}얼마{ques_trailing}',
            equation=gen.EqnRef('eqn08_2', length_k, n_k),

            env=gen.fnmap(
                sent_exp2=sent_exp2,
                sent_exp3=sent_exp3,
                length=length_k,
                n=n_k,

                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))


# @gen.problems.register
def prob08_2_7(selector, tokenpool, clskey):
    ''' 정삼각형의 한 변의 길이가 5cm일 때 둘레의 길이를 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    sent_exp2 = random.choice(['는 ', '의 길이는 '])
    sent_exp3 = random.choice(['한 ', ''])

    # new variables
    length = random.randint(1, 100)
    n = random.randint(1, 6)

    length_k = tokenpool.new(length)
    n_k = tokenpool.new(n)

    return gen.build(
            body='',
            question=f'정{gen.korutil.num2kor(n+3)}각형의 '\
            '{sent_exp3}변의 길이가 {length}cm일 때, '\
            '둘레의 길이를 {ques_trailing}',
            equation=gen.EqnRef('eqn08_2', length_k, n_k),

            env=gen.fnmap(
                sent_exp2=sent_exp2,
                sent_exp3=sent_exp3,
                length=length_k,
                n=n_k,

                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob08_2_8(selector, tokenpool, clskey):
    ''' 한 변의 길이가 5cm인 정삼각형 모양의 종이를 잘랐습니다. 
    이 종이의 둘레의 길이를 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    sent_exp2 = random.choice(['는 ', '의 길이는 '])
    sent_exp3 = random.choice(['한 ', ''])

    # new variables
    length = random.randint(1, 100)
    n = random.randint(1, 6)

    length_k = tokenpool.new(length)
    n_k = tokenpool.new(n)

    return gen.build(
            body='{sent_exp3}변의 길이가 {length}cm인 '\
            f'정{gen.korutil.num2kor(n+3)}각형'\
            '모양의 종이를 잘랐습니다.',
            question='이 종이의 둘레를 {ques_trailing}.',
            equation=gen.EqnRef('eqn08_2', length_k, n_k),

            env=gen.fnmap(
                sent_exp2=sent_exp2,
                sent_exp3=sent_exp3,
                length=length_k,
                n=n_k,

                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

def prob08_2_9(selector, tokenpool, clskey):
    ''' 정국이는 철사로 정삼각형을 만들었습니다. 
    한 변의 길이가 5cm일 때, 정삼각형의 둘레의 길이를 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['구하시오.', '구하세요.', '구해 보세요.'])
    sent_exp2 = random.choice(['는 ', '의 길이는 '])
    sent_exp3 = random.choice(['한 ', ''])

    # new variables
    length = random.randint(1, 100)
    n = random.randint(1, 6)
    name1 = selector.get(clskey.name)

    length_k = tokenpool.new(length)
    n_k = tokenpool.new(n)

    return gen.build(
            body='{name1}{#이?}는 철사로 '\
            f'정{gen.korutil.num2kor(n+3)}각형'\
            '을 만들었습니다.',
            question='{sent_exp3}변의 길이가 {length}cm일 때, '\
            f'정{gen.korutil.num2kor(n+3)}각형'\
            '의 둘레의 길이를 구하시오.',
            equation=gen.EqnRef('eqn08_2', length_k, n_k),

            env=gen.fnmap(
                sent_exp2=sent_exp2,
                sent_exp3=sent_exp3,
                length=length_k,
                n=n_k,
                name1=name1,

                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob08_2_10(selector, tokenpool, clskey):
    ''' 한 변의 길이가 5cm인 정삼각형 하나를 만들 때 필요한 철사의 길이는 몇 cm입니까? '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent_exp2 = random.choice(['는 ', '의 길이는 '])
    sent_exp3 = random.choice(['한 ', ''])

    # new variables
    length = random.randint(1, 100)
    n = random.randint(1, 6)
    name1 = selector.get(clskey.name)

    length_k = tokenpool.new(length)
    n_k = tokenpool.new(n)

    return gen.build(
            body='',
            question='{sent_exp3}변의 길이가 {length}cm인 '\
            f'정{gen.korutil.num2kor(n+3)}각형 '\
            '하나를 만들 때 필요한 철사의 길이는 몇 cm{ques_trailing}',
            equation=gen.EqnRef('eqn08_2', length_k, n_k),

            env=gen.fnmap(
                sent_exp2=sent_exp2,
                sent_exp3=sent_exp3,
                length=length_k,
                n=n_k,
                name1=name1,

                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

# @gen.problems.register
def prob08_2_11(selector, tokenpool, clskey):
    ''' 세 변의 길이가 모두 5cm인 정삼각형의 둘레의 길이를 구하시오. '''

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '일까요?', '인지 구하시오.', '인지 구하세요.', '인지 구해 보세요.', '인가?', '인가요?'])
    sent_exp2 = random.choice(['는 ', '의 길이는 '])
    sent_exp3 = random.choice(['한 ', ''])

    # new variables
    length = random.randint(1, 100)
    n = random.randint(1, 6)
    name1 = selector.get(clskey.name)

    length_k = tokenpool.new(length)
    n_k = tokenpool.new(n)

    return gen.build(
            body='',
            question='각 '\
            f'{gen.korutil.num2korunit(n+3)} 변의 '\
            '길이가 {length}cm인 '\
            f'정{gen.korutil.num2kor(n+3)}각형 '\
            '하나를 만들 때 필요한 철사의 길이는 몇 cm{ques_trailing}',
            equation=gen.EqnRef('eqn08_2', length_k, n_k),

            env=gen.fnmap(
                sent_exp2=sent_exp2,
                sent_exp3=sent_exp3,
                length=length_k,
                n=n_k,
                name1=name1,

                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))



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
