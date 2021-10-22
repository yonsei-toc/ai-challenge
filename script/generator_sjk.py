import generator.exports as gen

import functools
import itertools
import math
import random
import string


#@gen.problems.register
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

# @gen.problems.register
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

# @gen.problems.register
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

# @gen.problems.register
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

#@gen.problems.register
def prob4_01_05(selector, tokenpool, clskey):
    '''
    - 외숙모의 나이는. 외삼촌의 나이는, ... 가장 나이가 많은 사람의 나이에서 나이가 적은 사람의 나이를 뺀 수는?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    l    = random.randint(2,5)
    name = [ selector.get(clskey.female_family_relation) for _ in range(0,l)]
    nums = [ random.randint(0,100) for _ in range(0,l) ]

    ## name overlab check
    while len(set(name)) != len(name):
        name = [selector.get(clskey.female_family_relation) for _ in range(0, l)]
    nums_k = list(map(tokenpool.new, nums))

    # syntactic randomize
    ques_trailing = random.choice([' 수는 무엇입니까?', ' 수를 구하시오.'])
    cent_trailing = random.choice(['사람','가족'])

    body = ' '.join(['유나의 ',
           ', '.join('{' + 'name{}'.format(x) + '}' +
                     '의 나이는 {' + 'nums{}'.format(x)+'}살' for x in range(l)),
        '입니다.'
    ])

    envdict = {f'name{i}': name[i] for i in range(l)}
    envdict.update({f'nums{i}': nums_k[i] for i in range(l)})
    envdict['ques_trailing'] = ques_trailing
    envdict['cent_trailing'] = cent_trailing

    # print 포함
    return gen.build(
            # body is the background of problem settings
            body=body,
            question='가장 나이가 많은 {cent_trailing}의 나이에서 가장 나이가 적은 {cent_trailing}의 나이를 뺀{ques_trailing}?',
            equation=gen.EqnRef('maxSubmin', *nums_k),
            env=envdict)

#@gen.problems.register
def prob4_01_06(selector, tokenpool, clskey):
    '''
    - clskey.race에서 4명의 학생은 각각 ..점..점..점을 얻었습니다. 1등의 점수에서 꼴찌의 점수를 뺀 값은?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    l    = random.randint(2,5)
    nums = [ random.randint(0,100) for _ in range(0,l) ]
    race = selector.get(clskey.race)

    nums_k = list(map(tokenpool.new, nums))
    l_k    = tokenpool.new(l)

    # syntactic randomize
    ques_trailing = random.choice(['은 무엇입니까?', '을 구하시오.'])

    body = ''.join(['{race}에서 {l}명의 학생은 각각 ',
           ', '.join('{' + 'nums{}'.format(x) + '}점' for x in range(l)),
        '을 얻었습니다.'
    ])

    envdict = {f'nums{i}': nums_k[i] for i in range(l)}
    envdict['race'] = race
    envdict['l']    = l_k
    envdict['ques_trailing'] = ques_trailing

    # print 포함
    return gen.build(
            # body is the background of problem settings
            body=body,
            question='1등의 점수에서 {l}등의 점수를 뺀 값{ques_trailing}',
            equation=gen.EqnRef('maxSubmin', *nums_k),
            env=envdict)

# @gen.problems.register
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
# @gen.problems.register
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
            equation=gen.EqnRef('prob04_02_02', catL_k,*nums_k),
            env=envdict)

#@gen.problems.register
def prob04_02_03(selector, tokenpool, clskey):
    '''
    - {name, name, name}은 각각 cslkey.writable을 하나씩 뽑았다. 각 {writable}에 적힌 번호는 n.n.n 이다. 사람들이 뽑은 번호 중 k개를
    뽑아 만들 수 있는 가장 작은 수는?
    '''

    # this may be a meta-token (e.g., #1), referring a number.
    nums_len  = random.randint(2, 5)
    catL = random.randint(1, nums_len)

    nums_l   = [ i for i in range(0,9)]
    nums     = random.sample(nums_l, nums_len)
    names    = [ selector.get(clskey.name) for _ in range(0,len(nums_l))]
    writable = selector.get(clskey.writable)

    nums_k     = list(map(tokenpool.new, nums))
    nums_len_k = tokenpool.new(nums_len)
    catL_k     = tokenpool.new(catL)


    # syntactic randomize
    ques_trailing = random.choice(['는 무엇입니까?', '를 쓰시오.'])

    body = ' '.join([', '.join('{' + 'names{}'.format(x) + '}' for x in range(nums_len)) + '{#는} {writable}{#을} 하나씩 뽑았다.'
                    ,'각 {writable}에 적힌 번호는'
                    ,', '.join('{' + 'nums{}'.format(x) + '}' for x in range(nums_len)) + '이다.'
                     ])

    question = ' '.join(['사람들이 뽑은 번호 중 {catL}개를 뽑아 만들 수 있는',
                         f'{gen.korutil.num2korunit(catL)} 자리 수',
                         '중에서 가장 작은 수{ques_trailing}'])

    envdict = {f'nums{i}': nums_k[i] for i in range(0,nums_len)}
    envdict.update({f'names{i}': names[i] for i in range(0, nums_len)})
    envdict['nums_len']      = nums_len_k
    envdict['ques_trailing'] = ques_trailing
    envdict['catL']          = catL_k
    envdict['writable']      = writable

    # print 포함
    return gen.build(
            body=body,
            question=question,
            equation=gen.EqnRef('prob04_02', catL_k,*nums_k),
            env=envdict)

# @gen.problems.register
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

# @gen.problems.register
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

# @gen.problems.register
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
def prob04_03_05(selector, tokenpool, clskey):
    '''
floor, room room 운동장, 연습장, ... , n,n,n,n,n 층에 있다. 이중 k층/칸 보다 높은 층에 있는 room의 개수는?
    '''
    len_  = random.randint(1,5)
    nums  = [ random.randint(0,len_) for _ in range(0,len_) ]
    over  = random.randint(0,5)
    floor = selector.get(clskey.floor)
    field = [ selector.get(clskey.field) for _ in range(0, len_)]

    ## field overlab check
    while len(set(field)) != len(field):
        field = [selector.get(clskey.field) for _ in range(0, len_)]

    len_k  = tokenpool.new(len_)
    nums_k = list(map(tokenpool.new, nums))
    over_k = tokenpool.new(over)

    # syntactic randomize
    ques_trailing = random.choice(['에 있습니다.', ' 높이에 있습니다.'])
    cent_trailing = random.choice(['모두',''])

    envdict = {f'nums{i}': nums_k[i] for i in range(len_)}
    envdict.update({f'field{i}': field[i] for i in range(len_)})
    envdict['over']   = over_k
    envdict['len_']   = len_k
    envdict['floor']  = floor
    envdict['ques_trailing'] = ques_trailing
    envdict['cent_trailing'] = cent_trailing

    return gen.build(
            # body is the background of problem settings
            body=' '.join([
                ', '.join('{' + 'field{}'.format(x) + '}' + '{#이} '
                          + '{' + 'nums{}'.format(x) + '}' + '{floor}' for x in range(len_)) + '{ques_trailing}'
            ]),
            question='이 중에서 {over}{floor} 보다 높은 {floor}에 있는 곳은 {cent_trailing} 몇 곳 입니까?',
            equation=gen.EqnRef('prob04_03', over_k,*nums_k),
            env=envdict)

# @gen.problems.register
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


# @gen.problems.register
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

# @gen.problems.register
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

# @gen.problems.register
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
# @gen.problems.register
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

# @gen.problems.register
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

# @gen.problems.register
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
# @gen.problems.register
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

# @gen.problems.register
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

# @gen.problems.register
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

# @gen.problems.register
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

# @gen.problems.register
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


# @gen.problems.register
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


# @gen.problems.register
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


# @gen.problems.register
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
# @gen.problems.register
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


# @gen.problems.register
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

# @gen.problems.register
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

#@gen.problems.register
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


if __name__ == '__main__':
    class _Namespace():
        def __init__(self): pass

    with open('dict.json', 'rt', encoding='UTF8') as f:
        dictionary, clskey = gen.Dictionary.load(f.read())
    for fn in gen.problems:

        i = 0
        while i < 8:
            selector = gen.DictionarySelector(dictionary)
            tokenpool = gen.TokenPool()
            ret = fn(selector, tokenpool, clskey)
            if ret is not None:
                print(ret)
                i += 1
