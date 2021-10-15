import generator.exports as gen

import functools
import itertools
import math
import random
import string


@gen.problems.register
def prob4_01_01(selector, tokenpool, clskey):
    '''
    - 43, 92, 71, 64가 있습니다. 그중에서 가장 큰 수에서 가장 작은 수를 뺀 값은 얼마입니까?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    nums_len = random.randint(2, 5)
    nums_l   = [ i for i in range(0,100)]
    nums     = random.sample(nums_l, nums_len)
    nums_k   = list(map(tokenpool.new, nums))

    # syntactic randomize
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    envdict = {f'num{i}': nums_k[i] for i in range(nums_len)}
    envdict['ques_trailing'] = ques_trailing

    # print 포함
    return gen.build(
            # body is the background of problem settings
            body=' '.join([', '.join('{' + 'num{}'.format(x) + '}' for x in range(nums_len)), '가 있습니다.'
            ]),
            question='그중에서 가장 큰 수에서 가장 작은 수를 뺀 값은 얼마{ques_trailing}',
            equation=gen.EqnRef('maxSubmin', *nums_k),
            env=envdict)

@gen.problems.register
def prob4_01_02(selector, tokenpool, clskey):
    '''
    - 은지, 태형, 유나, ..은 각각 사과 43, 92, 71, 64를 가지고 있습니다. 가장 많은 사과를 가진사람의 개수에서 가장 작은 사과를 가진 사름의 개수를 뺀 수는?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    item = selector.get(clskey.fruit)
    nums_len = random.randint(2, 5)
    nums = [random.randint(0,100) for _ in range(0, nums_len)]
    name = [selector.get(clskey.name) for _ in range(0, nums_len)]
    unit_ = item.of('unit')

    ## name overlab checkunit
    while len(set(name)) != len(name):
        name = [selector.get(clskey.name) for _ in range(0, nums_len)]

    item_k     = tokenpool.new(item)
    nums_k     = list(map(tokenpool.new, nums))
    name_k     = list(map(tokenpool.new, name))
    unit_k     = tokenpool.new(unit_)

    # syntactic randomize
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    body = ' '.join([
        ' , '.join('{' + 'name{}'.format(x) + '}' for x in range(nums_len)),
        '은 각각 {item}{#를}',
        ', '.join('{' + 'nums{}'.format(x) + '}' for x in range(nums_len)),
        '{unit_}{#를} 가지고 있습니다.'
    ])

    envdict = {f'nums{i}': nums_k[i] for i in range(nums_len)}
    envdict.update({f'name{i}': name_k[i] for i in range(nums_len)})
    envdict['ques_trailing'] = ques_trailing
    envdict['item']  = item_k
    envdict['unit_'] = unit_k

    # print 포함
    return gen.build(
            # body is the background of problem settings
            body=body,
            question='많은 {item}{#를} 가진사람의 개수에서 가장 작은 {item}{#를} 가진 사람의 개수를 뺀 수는 몇 {unit_}입니까?',
            equation=gen.EqnRef('maxSubmin', *nums_k),
            env=envdict)

@gen.problems.register
def prob4_01_03(selector, tokenpool, clskey):
    '''
    - 은지, 태형, 유나, ..은 각각 주스를 43l, 92l, 71l, 64l 마셨습니다. 가장 많은 주수를 마신사람의 개수에서 가장 적게 주스를 마신 사람의 l를 뺀 수는?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    item = selector.get(clskey.drink)
    nums_len = random.randint(2, 5)
    nums = [random.randint(0,100) for _ in range(0, nums_len)]
    name = [selector.get(clskey.name) for _ in range(0, nums_len)]

    ## name overlab checkunit
    while len(set(name)) != len(name):
        name = [selector.get(clskey.name) for _ in range(0, nums_len)]

    item_k     = tokenpool.new(item)
    nums_k     = list(map(tokenpool.new, nums))
    name_k     = list(map(tokenpool.new, name))

    # syntactic randomize
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    body = ' '.join([
        ' , '.join('{' + 'name{}'.format(x) + '}' for x in range(nums_len)),
        '은 각각 {item}{#를}',
        ', '.join('{' + 'nums{}'.format(x) + '}l' for x in range(nums_len)),
        '를 마셨습니다.'
    ])

    envdict = {f'nums{i}': nums_k[i] for i in range(nums_len)}
    envdict.update({f'name{i}': name_k[i] for i in range(nums_len)})
    envdict['ques_trailing'] = ques_trailing
    envdict['item']  = item_k

    # print 포함
    return gen.build(
            # body is the background of problem settings
            body=body,
            question='많은 {item}{#를} 마신 사람의 l에서 가장 적은 {item}{#를} 마신 사람의 l를 뺀 수는 몇 l입니까?',
            equation=gen.EqnRef('maxSubmin', *nums_k),
            env=envdict)
@gen.problems.register
def prob4_01_04(selector, tokenpool, clskey):
    '''
    - 상자에 사탕이 각각 43, 92, 71, 64 개 들어있습니다. 사탕이 가장 많이 담긴 상자의 사탕수에서 ?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    container = selector.get(clskey.container)
    item = selector.get(clskey.flower)
    nums_len = random.randint(2, 5)
    nums = [random.randint(0,100) for _ in range(0, nums_len)]
    unit = item.of('unit')

    container_k  = tokenpool.new(container)
    item_k     = tokenpool.new(item)
    nums_k     = list(map(tokenpool.new, nums))
    unit_k     = tokenpool.new(unit)

    # syntactic randomize
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    body = ' '.join(['{container}에 {item}이 각각 ',
        ', '.join('{' + 'nums{}'.format(x) + '}{unit}' for x in range(nums_len)),
        '있습니다.'
    ])

    envdict = {f'nums{i}': nums_k[i] for i in range(nums_len)}
    envdict['ques_trailing'] = ques_trailing
    envdict['container'] = container_k
    envdict['unit']  = unit_k
    envdict['item']  = item_k

    # print 포함
    return gen.build(
            # body is the background of problem settings
            body=body,
            question='{item}{#이} 가장 많이 담긴 {item} 수에서 {item}{#이} 가장 적게 담긴 수를 뺀 수는 몇입니까?',
            equation=gen.EqnRef('maxSubmin', *nums_k),
            env=envdict)

@gen.problems.register
def prob04_02(selector, tokenpool, clskey):
    '''
    - 0, 3, 5, 6 중에서 서로 다른 숫자 3개를 뽑아 만들 수 있는 세 자리 수 중 에서 가장 작은 수를 쓰시오.
    '''

    # this may be a meta-token (e.g., #1), referring a number.
    nums_len  = random.randint(2, 5)
    nums_l   = [ i for i in range(0,9)]
    nums     = random.sample(nums_l, nums_len)
    catL     = random.randint(1,nums_len)

    nums_k     = list(map(tokenpool.new, nums))
    nums_len_k = tokenpool.new(nums_len)
    catL_k     = tokenpool.new(catL)

    # syntactic randomize
    ques_trailing = random.choice(['는 무엇입니까?', '를 쓰시오.'])

    question = ' '.join(['그 중에서 서로 다른 숫자 {catL}개를 뽑아 만들 수 있는',
                         f'{gen.korutil.num2korunit(catL)} 자리 수',
                         ' 중에서 가장 작은 수{ques_trailing}'])
    envdict = {f'num{i}': nums_k[i] for i in range(0,nums_len)}
    envdict['nums_len']      = nums_len_k
    envdict['ques_trailing'] = ques_trailing
    envdict['catL']          = catL_k

    # print 포함
    return gen.build(
            body=' '.join([', '.join('{' + 'num{}'.format(x) + '}' for x in range(nums_len)), '가 있습니다.'
            ]),
            question=question,
            equation=gen.EqnRef('prob04_02', catL_k,*nums_k),
            env=envdict)

# equation 과 같은 문제로 augment 힘들어서 eq를 추가함
@gen.problems.register
def prob04_02_02(selector, tokenpool, clskey):
    '''
    - 0, 3, 5, 6 중에서 서로 다른 숫자 3개를 뽑아 만들 수 있는 세 자리 수 중 에서 가장 큰 수를 쓰시오.
    '''

    # this may be a meta-token (e.g., #1), referring a number.
    nums_len  = random.randint(2, 5)
    nums_l   = [ i for i in range(0,9)]
    nums     = random.sample(nums_l, nums_len)
    catL     = random.randint(1,nums_len)

    nums_k     = list(map(tokenpool.new, nums))
    nums_len_k = tokenpool.new(nums_len)
    catL_k     = tokenpool.new(catL)

    # syntactic randomize
    ques_trailing = random.choice(['는 무엇입니까?', '를 쓰시오.'])

    question = ' '.join(['그 중에서 서로 다른 숫자 {catL}개를 뽑아 만들 수 있는',
                         f'{gen.korutil.num2korunit(catL)} 자리 수',
                         ' 중에서 가장 작은 수{ques_trailing}'])
    envdict = {f'num{i}': nums_k[i] for i in range(0,nums_len)}
    envdict['nums_len']      = nums_len_k
    envdict['ques_trailing'] = ques_trailing
    envdict['catL']          = catL_k

    # print 포함
    return gen.build(
            body=' '.join([', '.join('{' + 'num{}'.format(x) + '}' for x in range(nums_len)), '가 있습니다.'
            ]),
            question=question,
            equation=gen.EqnRef('prob04_02', catL_k,*nums_k),
            env=envdict)

@gen.problems.register
def prob04_03_01(selector, tokenpool, clskey):
    '''
    5개의 수 1.4, 9/10, 1, 0.5, 13/10이 있습니다.이 중에서 1보다 큰 수는 모두 몇 개입니까?
    ** eq param으로 list를 두개 줄 없어서 일단 / 분수 형태가 아닌 소수로 줌
    '''
    len_ = random.randint(0,6)
    nums = [ round(float(random.uniform(0,3)),1) for _ in range(0,len_) ]
    over = random.randint(0,2)

    len_k  = tokenpool.new(len_)
    nums_k = list(map(tokenpool.new, nums))
    over_k = tokenpool.new(over)

    envdict = {f'num{i}': nums_k[i] for i in range(len_)}
    envdict['over']   = over_k
    envdict['len_']   = len_k

    return gen.build(
            # body is the background of problem settings
            body=' '.join([
                '{len_}개의 수 ',
                ', '.join('{' + 'num{}'.format(x) + '}' for x in range(len_)),
                '가 있습니다.'
            ]),
            question='이 중에서 {over}보다 큰 수는 모두 몇 개입니까?',
            equation=gen.EqnRef('prob04_03', over_k,*nums_k),
            env=envdict)

@gen.problems.register
def prob04_03_02(selector, tokenpool, clskey):
    '''
    유나의 키는 156cm, ..의키는.. 입니다. 이중 160cm 보다 의 키보다 큰 사람은 모두 몇 명 입니까?
    '''

    len_  = random.randint(1,6)
    names = [selector.get(clskey.name) for _ in range(len_)]
    nums  = [ round(float(random.uniform(0,3)),1) for _ in range(0,len_) ]
    over  = random.randint(0,2)
    item  = selector.get(clskey.tool)

    len_k  = tokenpool.new(len_)
    nums_k = list(map(tokenpool.new, nums))
    over_k = tokenpool.new(over)
    names_k = list(map(tokenpool.new, names))
    item_k  = tokenpool.new(item)

    # syntactic randomize
    ques_trailing = random.choice(['인지 구하시오.', '입니까?'])

    envdict = {f'num{i}': nums_k[i] for i in range(len_)}
    envdict.update({f'name{i}': names_k[i] for i in range(len_)})
    envdict['over']   = over_k
    envdict['len_']   = len_k
    envdict['ques_trailing'] = ques_trailing
    envdict['item'] = item_k

    return gen.build(
            # body is the background of problem settings
            body=' '.join([
                ', '.join('{' + 'name{}'.format(x) + '}의 {item}{#은} ' + '{' + 'num{}'.format(x) + '}cm' for x in range(len_)),
                '입니다.'
            ]),
            question='이 중에서 {over}cm 보다 {item}{#가} 큰 사람은 모두 몇 명{ques_trailing}.',
            equation=gen.EqnRef('prob04_03', over_k,*nums_k),
            env=envdict)

@gen.problems.register
def prob04_03_03(selector, tokenpool, clskey):
    '''
    상자 4개의 무게는 .. , .. , .. , .. 입니다. 이 중 .. 단위 보다 무거운 상자의 개수는?
    '''
    container = selector.get(clskey.container)
    unit  = selector.get(clskey.weight_unit)
    len_  = random.randint(1,6)
    nums  = [ round(float(random.uniform(0,3)),1) for _ in range(0,len_) ]
    over  = random.randint(0,2)

    len_k  = tokenpool.new(len_)
    nums_k = list(map(tokenpool.new, nums))
    over_k = tokenpool.new(over)

    # syntactic randomize
    ques_trailing = random.choice(['인지 구하시오.', '입니까?'])

    envdict = {f'num{i}': nums_k[i] for i in range(len_)}
    envdict['container'] = container
    envdict['over']   = over_k
    envdict['len_']   = len_
    envdict['ques_trailing'] = ques_trailing
    envdict['unit'] = unit

    return gen.build(
            # body is the background of problem settings
            body=' '.join(['{container} {len_}개의 무게는 ',
                ', '.join('{' + 'num{}'.format(x) + '}{unit}' for x in range(len_)),
                '입니다.'
            ]),
            question='이 중에서 {over}{unit} 보다 무거운 {container}{#는} 모두 몇 개{ques_trailing}.',
            equation=gen.EqnRef('prob04_03', over_k,*nums_k),
            env=envdict)

@gen.problems.register
def prob04_03_04(selector, tokenpool, clskey):
    '''
    상자 4개의 무게는 .. , .. , .. , .. 입니다. 이 중 .. 단위 보다 무거운 상자의 개수는?
    '''
    item = selector.get(clskey.wire)
    unit = item.of('unit')
    unit_len  = selector.get(clskey.length_unit)
    len_  = random.randint(1,6)
    nums  = [ round(float(random.uniform(0,3)),1) for _ in range(0,len_) ]
    over  = random.randint(0,2)

    len_k  = tokenpool.new(len_)
    nums_k = list(map(tokenpool.new, nums))
    over_k = tokenpool.new(over)

    # syntactic randomize
    ques_trailing = random.choice(['인지 구하시오.', '입니까?'])

    envdict = {f'num{i}': nums_k[i] for i in range(len_)}
    envdict['over']   = over_k
    envdict['len_']   = len_
    envdict['ques_trailing'] = ques_trailing
    envdict['unit'] = unit
    envdict['unit_len'] = unit_len
    envdict['item'] = item

    return gen.build(
            # body is the background of problem settings
            body=' '.join(['{item} {len_}개의 길이는 ',
                ', '.join('{' + 'num{}'.format(x) + '}{unit_len}' for x in range(len_)),
                '입니다.'
            ]),
            question='이 중에서 {over}{unit_len} 보다 긴 {item}{#는} 모두 몇 개{ques_trailing}',
            equation=gen.EqnRef('prob04_03', over_k,*nums_k),
            env=envdict)


@gen.problems.register
def prob04_04(selector, tokenpool, clskey):
    '''
    유나가 책을 펼쳤는데 두 쪽수의 합이 125이었습니다. 유나가 펼친 두 쪽수 중 큰 수를 쓰시오.
    '''
    # Claim items at first. They will not overlap (even for different keys).
    name     = selector.get(clskey.name)
    count1   = random.randrange(1, 400, 2)
    count1_k = tokenpool.new(count1)

    # syntactic randomize
    sent_trailing = random.choice(['다.','습니다.'])
    ques_trailing = random.choice(['를 쓰시오', '는 무엇입니까?', '는 무엇인가?'])

    return gen.build(
            # body is the background of problem settings
            body=' '.join([
                '{name}{#가} 책을 펼쳤는데 두 쪽수의 합이 {count1} 이{sent_trailing}',
            ]),
            # question is the main sentence of the problem
            question='{name}{#가} 펼친 두 쪽수중 큰 수{ques_trailing}',
            equation= gen.EqnRef('halfOdd',count1_k),

            # specify every variables used in above strings
            env=gen.fnmap(
                name=name,
                count1=count1_k,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

@gen.problems.register
def prob04_04_02(selector, tokenpool, clskey):
    '''
    연속되는 자연수 두 수의 합이 n이다.  연속된 자연수 중 더 큰 수를 쓰시오.
    '''
    # Claim items at first. They will not overlap (even for different keys).
    count1   = random.randrange(1, 400, 2)
    count1_k = tokenpool.new(count1)

    # syntactic randomize
    sent_trailing = random.choice(['이다.','입니다'])
    ques_trailing = random.choice(['를 쓰시오', '는 무엇입니까?', '는 무엇인가?'])

    return gen.build(
            # body is the background of problem settings
            body=' '.join([
                '연속되는 자연수 두 수의 합이 {count1} {sent_trailing}',
            ]),
            # question is the main sentence of the problem
            question='연속되는 자연수 중 더 큰 수{ques_trailing}',
            equation= gen.EqnRef('halfOdd',count1_k),

            # specify every variables used in above strings
            env=gen.fnmap(
                count1=count1_k,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

@gen.problems.register
def prob04_04_03(selector, tokenpool, clskey):
    '''
    철수와 영희의 나이차는 한살 차이이며 두 사람의 나이의 합은 n이다. 두명 중 나이가 더 많은 사람의 나이를 쓰시오.
    '''
    # Claim items at first. They will not overlap (even for different keys).
    names    = [ selector.get(clskey.name) for _ in range(2)]
    count1   = random.randrange(1, 400, 2)
    count1_k = tokenpool.new(count1)

    # syntactic randomize
    ques_trailing = random.choice(['를 쓰시오', '는 무엇입니까?', '는 무엇인가?'])

    return gen.build(
            # body is the background of problem settings
            body=' '.join([
                '{name0}{#와} {name1}의 나이차는 한 살 차이이며 두 사람의 나이의 합은 {count1}이다.'
            ]),
            # question is the main sentence of the problem
            question='두명 중 나이가 더 많은 사람의 나이{ques_trailing}',
            equation= gen.EqnRef('halfOdd',count1_k),

            # specify every variables used in above strings
            env=gen.fnmap(
                count1=count1_k,
                ques_trailing=ques_trailing,
                name0=names[0],
                name1=names[1]
            ))
@gen.problems.register
def prob04_04_04(selector, tokenpool, clskey):
    '''
    외숙모와 아빠의 키차이는
    '''
    # Claim items at first. They will not overlap (even for different keys).
    names    = [ selector.get(clskey.common_family_relation) for _ in range(2)]
    count1   = random.randrange(1, 400, 2)
    count1_k = tokenpool.new(count1)
    #unit     = selector.get(clskey.length_unit)
    unit     = 'cm'
    # syntactic randomize
    ques_trailing = random.choice(['인지 쓰시오.', '입니까?'])

    return gen.build(
            # body is the background of problem settings
            body=' '.join([
                '{name0}{#와} {name1}의 키차이는 1{unit} 차이이며 두 사람의 키의 합은 {count1}{unit}이다.'
            ]),
            # question is the main sentence of the problem
            question='두명 중 키가 큰 사람의 키는 몇 {unit}{ques_trailing}',
            equation= gen.EqnRef('halfOdd',count1_k),

            # specify every variables used in above strings
            env=gen.fnmap(
                count1=count1_k,
                ques_trailing=ques_trailing,
                name0=names[0],
                name1=names[1],
                unit=unit
            ))

@gen.problems.register
def prob04_1_05(selector, tokenpool, clskey):
    '''
    어떤 소수의 소수점을 왼쪽으로 두 자리 옮기면 원래의 소수보다 1.782만큼 작아집니다. 원래의 소수를 구하시오.
    '''
    # Claim items at first. They will not overlap (even for different keys).
    n     = random.randint(0,3)
    count1   = round(float(random.random()),2) + random.randint(0,10)

    n_k   = tokenpool.new(n)
    count1_k = tokenpool.new(count1)

    # syntactic randomize
    sent_trailing = random.choice(['집니다.','졌습니다.'])
    ques_trailing = random.choice(['를 쓰시오', '는 무엇입니까?', '는 무엇인가?'])

    return gen.build(
            body=' '.join([
                f'어떤 소수의 소수점을 왼쪽으로 {gen.korutil.num2korunit(n)}', '자리 옮기면 원래의 소수보다 {count1} 만큼 작아{sent_trailing}',
            ]),
            question='원래의 소수{ques_trailing}',
            equation= gen.EqnRef('getDeci',count1_k, n_k),
            env=gen.fnmap(
                n= n_k,
                count1=count1_k,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing
            ))

@gen.problems.register
def prob06_04_01(selector, tokenpool, clskey):
    '''어떤 수에 14를 더한 후 14를 곱하고, 24를 뺀 값을 24로 나누면 13이 됩니다. 어떤 수를 소수자리에서 올림한 수를 구하시오.'''
    # Claim items at first. They will not overlap (even for different keys).
    count1   = random.randrange(1, 100)
    count2 = random.randrange(1, 100)
    count3 = random.randrange(1, 100)
    count1_k = tokenpool.new(count1)
    count2_k = tokenpool.new(count2)
    count3_k = tokenpool.new(count3)

    # syntactic randomize
    ques_trailing = random.choice(['를 쓰시오', '는 무엇입니까?', '는 무엇인가?'])

    return gen.build(
            body=' '.join([
                '어떤 수에 를 {count1}{#를} 더한 후 {count1}{#를} 곱하고, {count2}{#를} 뺀 값을 {count2}로 나누면 {count3}{#이} 됩니다.',
            ]),
            question='어떤 수를 소수자리에서 올림한 수{ques_trailing}',
            equation=gen.EqnRef('prob06_04', count1_k, count2_k, count3_k),
            env=gen.fnmap(
                count1=count1_k,
                count2=count2_k,
                count3=count3_k,
                ques_trailing= ques_trailing
            ))

@gen.problems.register
def prob06_04_02(selector, tokenpool, clskey):
    '''처음 수박을 갯수에 14를 더한 후 14를 곱하고, 24를 뺀 값을 24명이 똑같이 나눠가지면 한 사람당 13개 이상 가집니다.. 처음 수박의 개수는 최소 몇개인지 '''
    # Claim items at first. They will not overlap (even for different keys).
    fruit    = selector.get(clskey.fruit)
    count1   = random.randrange(1, 100)
    count2 = random.randrange(1, 100)
    count3 = random.randrange(1, 100)
    count1_k = tokenpool.new(count1)
    count2_k = tokenpool.new(count2)
    count3_k = tokenpool.new(count3)

    # syntactic randomize
    ques_trailing = random.choice(['를 쓰시오.', '구하시오.'])

    return gen.build(
            body=' '.join([
                '처음 {fruit}의 개수에  를 {count1}개를 더한 후 {count1}{#를} 곱하고, {count2}개를 뺀 값을 {count2}명이 나눠가지면 한 사람당 {count3}개 이상을 가집니다.',
            ]),
            question='처음 {fruit}의 개수는 최소 몇개인지{ques_trailing}',
            equation=gen.EqnRef('prob06_04', count1_k, count2_k, count3_k),
            env=gen.fnmap(
                count1=count1_k,
                count2=count2_k,
                count3=count3_k,
                fruit = fruit,
                ques_trailing= ques_trailing
            ))

@gen.problems.register
def prob06_04_03(selector, tokenpool, clskey):
    '''처음 container의 drink를 14l를 더한 후 14배로 불렸다, 24l를 뺀 값을 24{container}에 똑같이 나누면 한 container 당 13l 이상 가집니다. 처음 container의 l를 소수자리에서 올림할 때 몇 l인가?'''
    # Claim items at first. They will not overlap (even for different keys).
    container = selector.get(clskey.container)
    drink    = selector.get(clskey.drink)
    unit     = selector.get(clskey.volume_unit)
    count1   = random.randrange(1, 100)
    count2 = random.randrange(1, 100)
    count3 = random.randrange(1, 100)
    count1_k = tokenpool.new(count1)
    count2_k = tokenpool.new(count2)
    count3_k = tokenpool.new(count3)

    # syntactic randomize
    ques_trailing = random.choice(['쓰시오.', '구하시오.'])

    return gen.build(
            body=' '.join([
                '{container}에 {drink}{#를} {count1}{unit}{#를} 더한 후 {count1}배로 불렸다.',
                '그리고 {count2}{unit}{#을} 뺀 값을 {count2}{container}에 나누면 한 {container}에 {count3}{unit}{#이} 담깁니다.',
            ]),
            question='처음 {container}{#는} 몇 {unit}인지 소수 첫째자리에서 올린 값을 {ques_trailing}',
            equation=gen.EqnRef('prob06_04', count1_k, count2_k, count3_k),
            env=gen.fnmap(
                count1=count1_k,
                count2=count2_k,
                count3=count3_k,
                container = container,
                drink = drink,
                unit = unit,
                ques_trailing= ques_trailing
            ))

@gen.problems.register
def prob06_04_04(selector, tokenpool, clskey):
    '''처음 영희가 가지고 있는 {flow}에 14개를 더한 수의 14배의 {flow}{#가}있다. 그리고 24{unit}를 뺀 값을 24{container}명 똑같이 나누면 한 container 당 13l 이상 가집니다. 처음 container의 l를 소수자리에서 올림할 때 몇 l인가?'''
    # Claim items at first. They will not overlap (even for different keys).
    name     = selector.get(clskey.name)
    flower    = selector.get(clskey.flower)
    unit     = flower.of("unit")
    count1   = random.randrange(1, 100)
    count2 = random.randrange(1, 100)
    count3 = random.randrange(1, 100)
    count1_k = tokenpool.new(count1)
    count2_k = tokenpool.new(count2)
    count3_k = tokenpool.new(count3)

    # syntactic randomize
    ques_trailing = random.choice(['쓰시오.', '구하시오.'])

    return gen.build(
            body=' '.join([
                '처음 {name}가 가지고 있는 {flower}의 개수에 {count1}{unit}{#를} 더한 후 {count1}{#를} 곱한만큼 샀다. ',
                '그리고 {count2}{unit}{#을} 뺀 값을 {count2}명이 나누면 한 사람당 {flower}{#는} {count3}{unit}{#를} 가진다.',
            ]),
            question='처음 {flower}{#는} 최소 몇 {unit}인지 {ques_trailing}',
            equation=gen.EqnRef('prob06_04', count1_k, count2_k, count3_k),
            env=gen.fnmap(
                count1=count1_k,
                count2=count2_k,
                count3=count3_k,
                flower = flower,
                unit = unit,
                name = name,
                ques_trailing= ques_trailing
            ))

@gen.problems.register
def prob07_04(selector, tokenpool, clskey):
    '''
    !! name overlap !!
    지민이는 주스를 0.7l 마셨습니다. 은지는 지민이보다 1/10l 더 적게 마셨습니다. 윤기는 4/5l 마셨고, 유나는 지민이보다 0.2l 더 많이 마셨습니다. 주스를 가장 많이 마신사람은 누구입니까?
    '''
    item = selector.get(clskey.drink)
    unit = selector.get(clskey.volume_unit)
    name = [ selector.get(clskey.name) for _ in range(0,4)]
    nums = [ round(float(random.uniform(0,2)),1) for _ in range(0,4) ]

    ## name overlab check
    while len(set(name)) != len(name):
        name = [selector.get(clskey.name) for _ in range(0, 4)]

    item_k = tokenpool.new(item)
    unit_k = tokenpool.new(unit)
    name_k = list(map(tokenpool.new, name))
    nums_k = list(map(tokenpool.new, nums))

    envdict = {f'name{i}': name_k[i] for i in range(4)}
    envdict.update({f'nums{i}': nums_k[i] for i in range(4)})
    envdict['unit'] = unit_k
    envdict['item'] = item_k

    return gen.build(
            body=' '.join([
                '{name0}{#이}는 {item}{#를} {nums0}{unit} 마셨습니다.',
                '{name1}{#이}는 {name0}{#이}보다 {nums1}{unit} 더 적게 마셨습니다. ',
                '{name2}{#는} {nums2}{unit} 마셨고, {name3}{#는} {name0}{#이}보다 {nums3}{unit} 더 많이 마셨습니다.'
            ]),
            question='주스를 가장 많이 마신사람은 누구입니까?',
            equation=gen.EqnRef('prob07_04', name_k[0], name_k[1], name_k[2], name_k[3], *nums_k),
            env=envdict
    )


@gen.problems.register
def prob07_04_02(selector, tokenpool, clskey):
    '''
    !! name overlap !!
    {family_relation}의 철사는 0.7{length}_unit 입니다.. 은지는 지민이보다 1/10l 더 적게 마셨습니다. 윤기는 4/5l 마셨고, 유나는 지민이보다 0.2l 더 많이 마셨습니다. 주스를 가장 많이 마신사람은 누구입니까?
    '''
    item = selector.get(clskey.wire)
    unit = selector.get(clskey.volume_unit)
    name = [ selector.get(clskey.female_family_relation) for _ in range(0,4)]
    nums = [ round(float(random.uniform(0,2)),1) for _ in range(0,4) ]

    ## name overlab check
    while len(set(name)) != len(name):
        name = [selector.get(clskey.name) for _ in range(0, 4)]

    item_k = tokenpool.new(item)
    unit_k = tokenpool.new(unit)
    name_k = list(map(tokenpool.new, name))
    nums_k = list(map(tokenpool.new, nums))

    envdict = {f'name{i}': name_k[i] for i in range(4)}
    envdict.update({f'nums{i}': nums_k[i] for i in range(4)})
    envdict['unit'] = unit_k
    envdict['item'] = item_k

    return gen.build(
            body=' '.join([
                '{name0}{#는} {item}{#를} {nums0}{unit} 만큼 가지고 있습니다.',
                '{name1}{#는} {name0}{#이}보다 {nums1}{unit} 짧은 {item}을 가지고 있습니다. ',
                '{name2}{#는} {nums2}{unit}의 {item}을 가지고 있고, {name3}{#는} {name0}보다 {nums3}{unit} 더 긴 {item}을 가지고 있습니다.'
            ]),
            question='가장 긴 {item}을 가진 사람은 누구입니까?`',
            equation=gen.EqnRef('prob07_04', name_k[0], name_k[1], name_k[2], name_k[3], *nums_k),
            env=envdict
    )


@gen.problems.register
def prob07_04_03(selector, tokenpool, clskey):
    '''
    !! name overlap !!
    지민이는 돈을 0.7원 가졌습니다. 은지는 지민이보다 1/10원 더 적게 가지고 있습니다.
    윤기는 4/5원 이 있고, 유나는 지민이보다 0.2원 더 있습니다. 주스를 가장 많이 마신사람은 누구입니까?
    '''
    item = '돈'
    unit = '원'
    name = [ selector.get(clskey.female_family_relation) for _ in range(0,4)]
    nums = [ round(float(random.uniform(0,2)),1) for _ in range(0,4) ]

    ## name overlab check
    while len(set(name)) != len(name):
        name = [selector.get(clskey.name) for _ in range(0, 4)]

    item_k = tokenpool.new(item)
    unit_k = tokenpool.new(unit)
    name_k = list(map(tokenpool.new, name))
    nums_k = list(map(tokenpool.new, nums))

    envdict = {f'name{i}': name_k[i] for i in range(4)}
    envdict.update({f'nums{i}': nums_k[i] for i in range(4)})
    envdict['unit'] = unit_k
    envdict['item'] = item_k

    return gen.build(
            body=' '.join([
                '{name0}{#는} {item}{#을} {nums0}{unit} 가지고 있습니다.',
                '{name1}{#는} {name0}{#이}보다 {nums1}{unit} 더 적게 가지고 있습니다. ',
                '{name2}{#는} {nums2}{unit}을 가지고 있으며, {name3}{#는} {name0}보다 {nums3}{unit} 더 많은 {item}을 가지고 있습니다.'
            ]),
            question='가장 많은 {item}을 가진 사람은 누구입니까?`',
            equation=gen.EqnRef('prob07_04', name_k[0], name_k[1], name_k[2], name_k[3], *nums_k),
            env=envdict
    )


@gen.problems.register
def prob07_04_04(selector, tokenpool, clskey):
    '''
    !! name overlap !!
    {place}까지의 거리는 0.7{length_unig} 입니다. {place}는 {place}보다 1/10{length_unit} 더 가깝습니다.
    {place}는 4/5{length_unit} 이 있고, 유나는 지민이보다 0.2원 더 있습니다. 가장 가까운 곳은 어디입니까?
    '''
    unit = selector.get(clskey.length_unit)
    name = [ selector.get(clskey.place) for _ in range(0,4)]
    nums = [ round(float(random.uniform(0,2)),1) for _ in range(0,4) ]

    ## name overlab check
    while len(set(name)) != len(name):
        name = [selector.get(clskey.place) for _ in range(0, 4)]

    unit_k = tokenpool.new(unit)
    name_k = list(map(tokenpool.new, name))
    nums_k = list(map(tokenpool.new, nums))

    envdict = {f'name{i}': name_k[i] for i in range(4)}
    envdict.update({f'nums{i}': nums_k[i] for i in range(4)})
    envdict['unit'] = unit_k

    return gen.build(
            body=' '.join([
                '{name0}까지의 거리는 {nums0}{unit} 입니다.',
                '{name1}{#는} {name0}보다 {nums1}{unit} 더 가깝습니다.',
                '{name2}까지의 거리는 {nums2}{unit}이며, {name3}{#는} {name0}보다 {nums3}{unit} 더 멉니다.'
            ]),
            question='가장 먼 곳은 어디입니까?',
            equation=gen.EqnRef('prob07_04', name_k[0], name_k[1], name_k[2], name_k[3], *nums_k),
            env=envdict
    )
@gen.problems.register
def prob08_04(selector, tokenpool, clskey):
    '''한 변의 길이가 10cm인 정사각형과 둘레가 같은 정팔각형이 있습니다. 이 정팔각형의 한 변의 길이는 몇 cm인지 소수점 둘째자리까지 구하시오.'''
    # 한 / 두 /.. num2kororder는 한..두.. 아홉까지 있어 choice를 사용함
    edge     = random.choice(['한','두','세'])
    item1    = selector.get(clskey.jshape)
    item2    = selector.get(clskey.jshape)
    l      = random.randint(1,100)
    unit   = selector.get(clskey.length_unit)

    item1_k  = tokenpool.new(item1)
    item2_k  = tokenpool.new(item2)
    l_k      = tokenpool.new(l)
    unit_k   = tokenpool.new(unit)
    edge_k   = edge

    return gen.build(
            body=' '.join([
                '{edge} 변의 길이가 {l}{unit}인 {item1}과 둘레가 같은 {item2} 있습니다.',
            ]),
            question='이 {item2}의 한 변의 길이는 몇 {unit}인지 소수 둘째자리 까지 쓰시오.',
            equation=gen.EqnRef('prob08_04',item1_k, item2_k, l_k, edge_k),
            env=gen.fnmap(
                item1=item1_k,
                item2=item2_k,
                l=l_k,
                unit = unit_k,
                edge = edge_k
            ))


@gen.problems.register
def prob08_04_02(selector, tokenpool, clskey):
    '''한 변의 길이가 10cm인 정사각형과 둘레가 같은 정팔각형이 있습니다.
    이 정팔각형을 철사로 두릅니다.
     이때 한 변을 두르기 위해 필요한 철사는몇 cm인지 소수점 둘째자리까지 구하시오.'''
    # 한 / 두 /.. num2kororder는 한..두.. 아홉까지 있어 choice를 사용함
    edge     = random.choice(['한','두','세'])
    wire     = selector.get(clskey.wire)
    item1    = selector.get(clskey.jshape)
    item2    = selector.get(clskey.jshape)
    l      = random.randint(1,100)
    unit   = selector.get(clskey.length_unit)

    item1_k  = tokenpool.new(item1)
    item2_k  = tokenpool.new(item2)
    l_k      = tokenpool.new(l)
    unit_k   = tokenpool.new(unit)
    edge_k   = edge

    return gen.build(
            body=' '.join([
                '{edge} 변의 길이가 {l}{unit}인 {item1}과 둘레가 같은 {item2}{#이} 있습니다.',
                '이 {item2}{#를} {wire}로 두릅니다.'
            ]),
            question='이 때 한 변을 두르기 위해 필요한 {wire}{#는} 몇 {unit}인지 소수 둘째자리 까지 쓰시오.',
            equation=gen.EqnRef('prob08_04',item1_k, item2_k, l_k, edge_k),
            env=gen.fnmap(
                item1=item1_k,
                item2=item2_k,
                l=l_k,
                unit = unit_k,
                edge = edge_k,
                wire = wire
            ))

@gen.problems.register
def prob08_04_03(selector, tokenpool, clskey):
    '''한 변의 길이가 10cm인 정사각형 종이과 둘레가 같은 정팔각형 종이가 있습니다.
     이때 종이의 한 변은 몇 {length unit}인지 소수점 둘째자리까지 구하시오.'''
    # 한 / 두 /.. num2kororder는 한..두.. 아홉까지 있어 choice를 사용함
    edge     = random.choice(['한','두','세'])
    side     = selector.get(clskey.side)
    item1    = selector.get(clskey.jshape)
    item2    = selector.get(clskey.jshape)
    l      = random.randint(1,100)
    unit   = selector.get(clskey.length_unit)

    item1_k  = tokenpool.new(item1)
    item2_k  = tokenpool.new(item2)
    l_k      = tokenpool.new(l)
    unit_k   = tokenpool.new(unit)
    edge_k   = edge

    return gen.build(
            body=' '.join([
                '{edge} 변의 길이가 {l}{unit}인 {side}{#와} 둘레가 같은 {item2} {side}{#가} 있습니다.',
            ]),
            question='이 때 {item2} {side}의 한 변의 길이는 몇 {unit}인지 소수 둘째자리 까지 쓰시오.',
            equation=gen.EqnRef('prob08_04',item1_k, item2_k, l_k, edge_k),
            env=gen.fnmap(
                item1=item1_k,
                item2=item2_k,
                l=l_k,
                unit = unit_k,
                edge = edge_k,
                side = side
            ))

@gen.problems.register
def prob08_04_04(selector, tokenpool, clskey):
    '''한 변의 길이가 10cm인 정사각형 운동장과 둘레가 같은 정팔각형 종이가 있습니다.
     이때 종이의 한 변은 몇 {length}인지 소수점 둘째자리까지 구하시오.'''
    # 한 / 두 /.. num2kororder는 한..두.. 아홉까지 있어 choice를 사용함
    edge     = random.choice(['한','두','세'])
    field     = selector.get(clskey.field)
    item1    = selector.get(clskey.jshape)
    item2    = selector.get(clskey.jshape)
    l      = random.randint(1,100)
    unit   = selector.get(clskey.length_unit)

    item1_k  = tokenpool.new(item1)
    item2_k  = tokenpool.new(item2)
    l_k      = tokenpool.new(l)
    unit_k   = tokenpool.new(unit)
    edge_k   = edge

    return gen.build(
            body=' '.join([
                '{edge} 변의 길이가 {l}{unit}인 {field}{#와} 둘레가 같은 {item2} {field}{#이} 있습니다.',
            ]),
            question='이 때 {item2} {field}의 한 변의 길이는 몇 {unit}인지 소수 둘째자리 까지 쓰시오.',
            equation=gen.EqnRef('prob08_04',item1_k, item2_k, l_k, edge_k),
            env=gen.fnmap(
                item1=item1_k,
                item2=item2_k,
                l=l_k,
                unit = unit_k,
                edge = edge_k,
                field = field
            ))

def build_dictionary(clskey, dictionary):
    # add mappings; use distinct keys!
    clskey.container = 'entity.container'
    clskey.fruit = 'entity.fruit'
    clskey.animal = 'entity.animal'
    clskey.item = 'entity.item'
    clskey.ride = 'entity.ride'
    clskey.flower = 'entity.flower'

    clskey.tool = 'entity.tool'
    clskey.tool_group = 'entity.tool.group'
    clskey.wire = 'entity.tool.wire'

    clskey.place = 'entity.place'
    clskey.name = 'entity.name'
    clskey.group = 'entity.group'

    clskey.color = 'prop.color'

    clskey.family_relation = 'prop.relation.family'
    clskey.male_family_relation = 'prop.relation.family.male'
    clskey.female_family_relation = 'prop.relation.family.female'
    clskey.common_family_relation = 'prop.relation.family.common'

    clskey.alpha = 'entity.alpha'
    clskey.edge = 'entity.edge'

    clskey.liquid   = 'entity.liquid'
    clskey.drink    = 'entity.drink'

    #clskey.num     = 'entity.num'

    # groups
    clskey.fruit_group = 'group.fruit'
    clskey.animal_group = 'group.animal'
    clskey.flower_group = 'group.flower'
    clskey.school_group = 'gruop.school'
    clskey.gender_group = 'group.gender'
    clskey.age_group = 'group.age'

    clskey.school = 'entity.school'
    clskey.gender = 'prop.gender'
    clskey.age = 'prop.age'


    clskey.length_unit = 'prop.unit.length'
    clskey.area_unit = 'prop.unit.area'
    clskey.volume_unit = 'prop.unit.volume'
    clskey.weight_unit = 'prop.unit.weight'

    clskey.ord_rel = 'prop.ord_rel'

    # setup hierarchy
    # item is a supertype of fruit, etc., etc..
    dictionary.set_child_relation(clskey.item, clskey.fruit)
    dictionary.set_child_relation(clskey.item, clskey.tool)
    dictionary.set_child_relation(clskey.item, clskey.container)
    dictionary.set_child_relation(clskey.item, clskey.flower)

    dictionary.set_child_relation(clskey.place, clskey.school_group)

    dictionary.set_child_relation(clskey.group, clskey.school_group)
    dictionary.set_child_relation(clskey.group, clskey.age_group)
    dictionary.set_child_relation(clskey.group, clskey.gender_group)
    dictionary.set_child_relation(clskey.group, clskey.fruit_group)
    dictionary.set_child_relation(clskey.group, clskey.animal_group)
    dictionary.set_child_relation(clskey.group, clskey.flower_group)

    dictionary.set_child_relation(clskey.tool, clskey.tool_group)
    dictionary.set_child_relation(clskey.tool, clskey.wire)

    dictionary.set_child_relation(clskey.family_relation, clskey.male_family_relation)
    dictionary.set_child_relation(clskey.family_relation, clskey.female_family_relation)
    dictionary.set_child_relation(clskey.male_family_relation, clskey.common_family_relation)
    dictionary.set_child_relation(clskey.female_family_relation, clskey.common_family_relation)
    dictionary.set_child_relation(clskey.liquid, clskey.drink)

    # add tokens
    dictionary.add_token(clskey.container, gen.DictItem('바구니', unit='개'))
    dictionary.add_token(clskey.container, gen.DictItem('상자', unit='개'))
    # dictionary.add_token(clskey.container, gen.DictItem('선반', unit='개'))
    dictionary.add_token(clskey.container, gen.DictItem('소쿠리', unit='개'))
    dictionary.add_token(clskey.container, gen.DictItem('그릇', unit='개'))
    dictionary.add_token(clskey.container, gen.DictItem('접시', unit='개'))
    dictionary.add_token(clskey.container, gen.DictItem('주머니', unit='개'))

    dictionary.add_token(clskey.fruit_group, gen.DictItem('과일', subkey=clskey.fruit))
    dictionary.add_token(clskey.fruit, gen.DictItem('사과', unit='개'))
    dictionary.add_token(clskey.fruit, gen.DictItem('배', unit='개'))
    dictionary.add_token(clskey.fruit, gen.DictItem('감', unit='개'))
    dictionary.add_token(clskey.fruit, gen.DictItem('귤', unit='개'))
    dictionary.add_token(clskey.fruit, gen.DictItem('포도', unit='송이'))
    dictionary.add_token(clskey.fruit, gen.DictItem('수박', unit='통'))
    dictionary.add_token(clskey.fruit, gen.DictItem('키위', unit='개'))
    dictionary.add_token(clskey.fruit, gen.DictItem('바나나', unit='개'))
    # dictionary.add_token(clskey.fruit, gen.DictItem('마카다미아', unit='개'))

    dictionary.add_token(clskey.animal_group, gen.DictItem('동물', subkey=clskey.animal))
    dictionary.add_token(clskey.animal, gen.DictItem('개', unit='마리'))
    dictionary.add_token(clskey.animal, gen.DictItem('고양이', unit='마리'))
    dictionary.add_token(clskey.animal, gen.DictItem('소', unit='마리'))
    dictionary.add_token(clskey.animal, gen.DictItem('말', unit='마리'))
    dictionary.add_token(clskey.animal, gen.DictItem('오리', unit='마리'))
    dictionary.add_token(clskey.animal, gen.DictItem('닭', unit='마리'))
    dictionary.add_token(clskey.animal, gen.DictItem('토끼', unit='마리'))
    dictionary.add_token(clskey.animal, gen.DictItem('물고기', unit='마리'))
    dictionary.add_token(clskey.animal, gen.DictItem('고래', unit='마리'))
    dictionary.add_token(clskey.animal, gen.DictItem('거위', unit='마리'))
    dictionary.add_token(clskey.animal, gen.DictItem('달팽이', unit='마리'))
    dictionary.add_token(clskey.animal, gen.DictItem('개구리', unit='마리'))
    # ...
    # dictionary.add_token(clskey.animal, '강아지', unit='마리')
    # dictionary.add_token(clskey.animal, '송아지', unit='마리')
    # dictionary.add_token(clskey.animal, '망아지', unit='마리')
    # dictionary.add_token(clskey.animal, '병아리', unit='마리')
    # ...

    dictionary.add_token(clskey.flower_group, gen.DictItem('꽃', subkey=clskey.flower))
    dictionary.add_token(clskey.flower, gen.DictItem('장미', unit='송이'))
    dictionary.add_token(clskey.flower, gen.DictItem('백합', unit='송이'))
    dictionary.add_token(clskey.flower, gen.DictItem('튤립', unit='송이'))
    dictionary.add_token(clskey.flower, gen.DictItem('카네이션', unit='송이'))
    dictionary.add_token(clskey.flower, gen.DictItem('국화', unit='송이'))
    dictionary.add_token(clskey.flower, gen.DictItem('화분', unit='송이'))
    dictionary.add_token(clskey.flower, gen.DictItem('화단', unit='송이'))

    dictionary.add_token(clskey.tool, gen.DictItem('책', unit='권'))
    dictionary.add_token(clskey.tool, gen.DictItem('공책', unit='권'))
    dictionary.add_token(clskey.tool, gen.DictItem('종이', unit='장'))
    dictionary.add_token(clskey.tool, gen.DictItem('도화지', unit='장'))
    dictionary.add_token(clskey.tool, gen.DictItem('색종이', unit='장'))
    dictionary.add_token(clskey.tool, gen.DictItem('지우개', unit='개'))
    dictionary.add_token(clskey.tool, gen.DictItem('컵', unit='개'))
    dictionary.add_token(clskey.tool, gen.DictItem('신발', unit='켤레'))
    dictionary.add_token(clskey.tool, gen.DictItem('꽃병', unit='병'))
    dictionary.add_token(clskey.tool, gen.DictItem('배구공', unit='개'))
    dictionary.add_token(clskey.tool, gen.DictItem('농구공', unit='개'))
    dictionary.add_token(clskey.tool, gen.DictItem('축구공', unit='개'))
    dictionary.add_token(clskey.tool, gen.DictItem('탁구공', unit='개'))
    dictionary.add_token(clskey.tool, gen.DictItem('야구공', unit='개'))

    # dictionary.add_token(clskey.tool, gen.DictItem('차', unit='대'))
    dictionary.add_token(clskey.ride, gen.DictItem('자동차', unit='대'))
    dictionary.add_token(clskey.ride, gen.DictItem('비행기', unit='대'))
    dictionary.add_token(clskey.ride, gen.DictItem('배', unit='척'))
    dictionary.add_token(clskey.ride, gen.DictItem('트럭', unit='대'))
    dictionary.add_token(clskey.ride, gen.DictItem('자전거', unit='대'))
    dictionary.add_token(clskey.ride, gen.DictItem('오토바이', unit='대'))
    dictionary.add_token(clskey.ride, gen.DictItem('기차', unit='대'))
    dictionary.add_token(clskey.ride, gen.DictItem('버스', unit='대'))
    dictionary.add_token(clskey.ride, gen.DictItem('엘리베이터', unit='대'))

    dictionary.add_token(clskey.color, gen.DictItem('흰', alias=['하얀']))
    dictionary.add_token(clskey.color, gen.DictItem('검은'))
    dictionary.add_token(clskey.color, gen.DictItem('빨간'))
    dictionary.add_token(clskey.color, gen.DictItem('노란'))
    dictionary.add_token(clskey.color, gen.DictItem('초록'))
    dictionary.add_token(clskey.color, gen.DictItem('파란'))
    dictionary.add_token(clskey.color, gen.DictItem('보라'))

    dictionary.add_token(clskey.tool_group, gen.DictItem('연필', unit='자루', group_unit=[('다스', 12)]))
    dictionary.add_token(clskey.tool_group, gen.DictItem('달걀', unit='알', group_unit=[('판', 30)]))

    dictionary.add_token(clskey.wire, gen.DictItem('철사', unit='개'))
    dictionary.add_token(clskey.wire, gen.DictItem('끈', unit='개'))
    dictionary.add_token(clskey.wire, gen.DictItem('줄', unit='개'))
    dictionary.add_token(clskey.wire, gen.DictItem('실', unit='개'))
    dictionary.add_token(clskey.wire, gen.DictItem('밧줄', unit='개'))
    dictionary.add_token(clskey.wire, gen.DictItem('전선', unit='개'))

    dictionary.add_token(clskey.place, gen.DictItem('서점'))
    dictionary.add_token(clskey.place, gen.DictItem('마트'))
    dictionary.add_token(clskey.place, gen.DictItem('문구점'))
    dictionary.add_token(clskey.place, gen.DictItem('집'))
    dictionary.add_token(clskey.place, gen.DictItem('수영장'))
    dictionary.add_token(clskey.place, gen.DictItem('교실'))
    dictionary.add_token(clskey.place, gen.DictItem('도서관'))
    dictionary.add_token(clskey.place, gen.DictItem('박물관'))
    dictionary.add_token(clskey.place, gen.DictItem('운동장'))
    dictionary.add_token(clskey.place, gen.DictItem('주차장'))
    dictionary.add_token(clskey.place, gen.DictItem('정류장'))
    dictionary.add_token(clskey.place, gen.DictItem('아파트'))
    dictionary.add_token(clskey.place, gen.DictItem('농장'))
    dictionary.add_token(clskey.place, gen.DictItem('강당'))

    dictionary.add_token(clskey.school_group, gen.DictItem('학교', subkey=clskey.school))
    dictionary.add_token(clskey.school, gen.DictItem('초등학교'))
    dictionary.add_token(clskey.school, gen.DictItem('중학교'))
    dictionary.add_token(clskey.school, gen.DictItem('고등학교'))

    # 공통 가족관계
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('할아버지'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('할머니'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('외할아버지'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('외할머니'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('고모부'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('고모'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('백부'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('백모'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('숙부'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('숙모'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('외숙모'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('외삼촌'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('이모부'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('이모'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('아버지'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('어머니'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('사촌'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('외사촌'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('조카'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('아들'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('며느리'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('손자'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('손녀'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('사위'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('딸'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('외손자'))
    dictionary.add_token(clskey.common_family_relation, gen.DictItem('외손녀'))
    # 남자 가족관계
    dictionary.add_token(clskey.male_family_relation, gen.DictItem('형'))
    dictionary.add_token(clskey.male_family_relation, gen.DictItem('누나'))
    dictionary.add_token(clskey.male_family_relation, gen.DictItem('형님'))
    dictionary.add_token(clskey.male_family_relation, gen.DictItem('처남'))
    dictionary.add_token(clskey.male_family_relation, gen.DictItem('아주머니'))
    dictionary.add_token(clskey.male_family_relation, gen.DictItem('처남의 댁'))
    dictionary.add_token(clskey.male_family_relation, gen.DictItem('처형'))
    dictionary.add_token(clskey.male_family_relation, gen.DictItem('동서'))
    dictionary.add_token(clskey.male_family_relation, gen.DictItem('처제'))
    # 여자 가족관계
    dictionary.add_token(clskey.female_family_relation, gen.DictItem('오빠'))
    dictionary.add_token(clskey.female_family_relation, gen.DictItem('언니'))
    dictionary.add_token(clskey.female_family_relation, gen.DictItem('아주버님'))
    dictionary.add_token(clskey.female_family_relation, gen.DictItem('형님'))
    dictionary.add_token(clskey.female_family_relation, gen.DictItem('도련님'))
    dictionary.add_token(clskey.female_family_relation, gen.DictItem('서방님'))
    dictionary.add_token(clskey.female_family_relation, gen.DictItem('동서'))
    dictionary.add_token(clskey.female_family_relation, gen.DictItem('아가씨'))

    # 이름에 가족관계 추가
    dictionary.add_token(clskey.name, gen.DictItem('남준', gender='male', family_relation=clskey.male_family_relation))
    dictionary.add_token(clskey.name, gen.DictItem('석진', gender='male', family_relation=clskey.male_family_relation))
    dictionary.add_token(clskey.name, gen.DictItem('윤기', gender='male', family_relation=clskey.male_family_relation))
    dictionary.add_token(clskey.name, gen.DictItem('호석', gender='male', family_relation=clskey.male_family_relation))
    dictionary.add_token(clskey.name, gen.DictItem('지민', gender='female', family_relation=clskey.female_family_relation))
    dictionary.add_token(clskey.name, gen.DictItem('태형', gender='male', family_relation=clskey.male_family_relation))
    dictionary.add_token(clskey.name, gen.DictItem('정국', gender='male', family_relation=clskey.male_family_relation))
    dictionary.add_token(clskey.name, gen.DictItem('민영', gender='female', family_relation=clskey.female_family_relation))
    dictionary.add_token(clskey.name, gen.DictItem('유정', gender='female', family_relation=clskey.female_family_relation))
    dictionary.add_token(clskey.name, gen.DictItem('은지', gender='female', family_relation=clskey.female_family_relation))
    dictionary.add_token(clskey.name, gen.DictItem('유나', gender='female', family_relation=clskey.female_family_relation))
    dictionary.add_token(clskey.name, gen.DictItem('철수', gender='male', family_relation=clskey.male_family_relation))
    dictionary.add_token(clskey.name, gen.DictItem('영희', gender='female', family_relation=clskey.female_family_relation))
    dictionary.add_token(clskey.name, gen.DictItem('영수', gender='male', family_relation=clskey.male_family_relation))

    dictionary.add_token(clskey.gender_group, gen.DictItem("성별", subkey=clskey.gender))
    dictionary.add_token(clskey.gender, gen.DictItem("남"))
    dictionary.add_token(clskey.gender, gen.DictItem("여"))

    dictionary.add_token(clskey.age_group, gen.DictItem("세대", subkey=clskey.age))
    dictionary.add_token(clskey.age, gen.DictItem("유년기"))
    dictionary.add_token(clskey.age, gen.DictItem("청소년기"))
    dictionary.add_token(clskey.age, gen.DictItem("성인기"))
    dictionary.add_token(clskey.age, gen.DictItem("노년기"))

    for i in string.ascii_uppercase:
        dictionary.add_token(clskey.alpha, gen.DictItem(str(i)))

    dictionary.add_token(clskey.edge, gen.DictItem('가로'))
    dictionary.add_token(clskey.edge, gen.DictItem('세로'))

    # SI 단위계는 직접 붙이기 바람
    dictionary.add_token(clskey.drink, gen.DictItem('물'))
    dictionary.add_token(clskey.drink, gen.DictItem('소금물'))
    dictionary.add_token(clskey.drink, gen.DictItem('주스'))
    dictionary.add_token(clskey.liquid, gen.DictItem('기름'))
    dictionary.add_token(clskey.liquid, gen.DictItem('간장'))
    dictionary.add_token(clskey.liquid, gen.DictItem('수도물'))
    dictionary.add_token(clskey.liquid, gen.DictItem('알코올'))

    dictionary.add_token(clskey.length_unit, gen.DictItem('km', factor=1000, kor="킬로미터", symbol="㎞"))
    dictionary.add_token(clskey.length_unit, gen.DictItem('m', factor=1, kor="미터", symbol="m"))
    dictionary.add_token(clskey.length_unit, gen.DictItem('cm', factor=0.01, kor="센티미터", symbol="㎝"))
    dictionary.add_token(clskey.length_unit, gen.DictItem('mm', factor=0.001, kor="밀리미터", symbol="㎜"))

    dictionary.add_token(clskey.volume_unit, gen.DictItem('kl', factor=1000, kor="킬로리터", symbol="㎘"))
    dictionary.add_token(clskey.volume_unit, gen.DictItem('l', factor=1, kor="리터", symbol="ℓ")	)
    dictionary.add_token(clskey.volume_unit, gen.DictItem('ml', factor=0.001, kor="밀리리터", symbol="㎖"))
    dictionary.add_token(clskey.volume_unit, gen.DictItem('m3', factor=1000, kor="세제곱미터", symbol="㎥"))
    dictionary.add_token(clskey.volume_unit, gen.DictItem('cm3', factor=0.001, kor="세제곱센티미터", symbol="㎤"))

    dictionary.add_token(clskey.weight_unit, gen.DictItem('t', factor=1000000, kor="톤", symbol="t"))
    dictionary.add_token(clskey.weight_unit, gen.DictItem('kg', factor=1000, kor="킬로그램", symbol="㎏"))
    dictionary.add_token(clskey.weight_unit, gen.DictItem('g', factor=1, kor="그램", symbol="g"))
    dictionary.add_token(clskey.weight_unit, gen.DictItem('mg', factor=0.001, kor="밀리그램", symbol="㎎"))

    dictionary.add_token(clskey.area_unit, gen.DictItem('km2', factor=1000000, kor="제곱킬로미터", symbol="㎢"))
    dictionary.add_token(clskey.area_unit, gen.DictItem('m2', factor=1, kor="제곱미터", symbol="㎡"))
    dictionary.add_token(clskey.area_unit, gen.DictItem('cm2', factor=0.0001, kor="제곱센티미터", symbol="㎠"))

    dictionary.add_token(clskey.ord_rel, gen.DictItem('크', reverse='작', var='큽니', reverse_var='작습니', adv='큰', reverse_adv='작은'))
    dictionary.add_token(clskey.ord_rel, gen.DictItem('무겁', reverse='가볍', var='무겁습니', reverse_var='가볍습니', adv='무거운', reverse_adv='가벼운'))
    dictionary.add_token(clskey.ord_rel, gen.DictItem('빠르', reverse='느리', var='빠릅니', reverse_var='느립니', adv='빠른', reverse_adv='느린'))
    # dictionary.add_token(clskey.ord_rel, gen.DictItem('높', reverse='낮', var='높습니', reverse_var='낮습니', adv='높은', reverse_adv='낮은'))
    # dictionary.add_token(clskey.ord_rel, gen.DictItem('많', reverse='적', var='많습니', reverse_var='적습니', adv='많은', reverse_adv='적은'))
    # dictionary.add_token(clskey.ord_rel, gen.DictItem('길', reverse='짧', var='깁니', reverse_var='짧습니', adv='긴', reverse_adv='짧은'))

    # dictionary.add_token(clskey.ord_rel, gen.DictItem('오른', reverse=['왼']))
    # dictionary.add_token(clskey.ord_rel, gen.DictItem('앞', reverse=['뒤']))

    ######## ADD MAPPINGS ########
    clskey.jshape       = 'entity.jshape'
    clskey.side = 'entity.side'
    clskey.field = 'entity.field'

    ######## ADJUST HIERARCHY ########
    dictionary.set_child_relation(clskey.tool, clskey.side)
    dictionary.set_child_relation(clskey.place, clskey.field)


    ######## ADD TOKENS ########
    dictionary.add_token(clskey.jshape, gen.DictItem('정사각형'))
    dictionary.add_token(clskey.jshape, gen.DictItem('정삼각형'))
    dictionary.add_token(clskey.jshape, gen.DictItem('정팔각형'))
    dictionary.add_token(clskey.jshape, gen.DictItem('정오각형'))


    dictionary.add_token(clskey.side, gen.DictItem('종이'))
    dictionary.add_token(clskey.side, gen.DictItem('포장지'))

    dictionary.add_token(clskey.field, gen.DictItem('경기장'))
    dictionary.add_token(clskey.field, gen.DictItem('운동장'))
    dictionary.add_token(clskey.field, gen.DictItem('극장'))




if __name__ == '__main__':
    class _Namespace():
        def __init__(self): pass


    clskey = _Namespace()
    dictionary = gen.Dictionary()
    build_dictionary(clskey, dictionary)
    for fn in gen.problems:
        i = 0
        while i < 8:
            selector = gen.DictionarySelector(dictionary)
            tokenpool = gen.TokenPool()
            ret = fn(selector, tokenpool, clskey)
            if ret is not None:
                print(ret)
                i += 1
