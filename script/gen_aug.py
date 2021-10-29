import generator.exports as gen
import random


@gen.problems.register
def prob4_01_01sum(selector, tokenpool, clskey):
    '''
    - 43, 92, 71, 64가 있습니다. 그중에서 가장 큰 수에서 가장 작은 수를 뺀 값은 얼마입니까?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    nums_len = random.randint(2, 5)
    nums_k = tokenpool.sample(range(0, 100), nums_len)

    envdict = {f'nums': nums_k}

    # syntactic randomize
    body_trailing = random.choice(['있습니다.', '있다.', '나열되어 있다.', '나열되어 있습니다.'])
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    envdict['ques_trailing'] = ques_trailing

    # print 포함
    return gen.build(
        # body is the background of problem settings
        body='{nums}가 ' + body_trailing,
        question='이 수들의 합계는 얼마{ques_trailing}',
        equation=gen.EqnRef('extract_from_nums', nums_k, 4),
        env=envdict)


@gen.problems.register
def prob4_01_02sum(selector, tokenpool, clskey):
    '''
    - 은지, 태형, 유나, ..은 각각 사과 43, 92, 71, 64를 가지고 있습니다. 가장 많은 사과를 가진사람의 개수에서 가장 작은 사과를 가진 사름의 개수를 뺀 수는?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    item = selector.get(clskey.fruit)
    nums_len = random.randint(2, 5)
    nums_k = tokenpool.sample(range(0, 100), nums_len)

    name = [selector.get(clskey.name) for _ in range(0, nums_len)]
    unit_ = item.of('unit')

    ## name overlab checkunit
    while len(set(name)) != len(name):
        name = [selector.get(clskey.name) for _ in range(0, nums_len)]

    # syntactic randomize
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    body = ' '.join([
        ' , '.join('{' + 'name{}'.format(x) + '}' for x in range(nums_len)),
        '은 각각 {item}{#를}',
        '{nums}',
        '{unit_}{#를} 가지고 있습니다.'
    ])

    envdict = {f'nums': nums_k}
    envdict.update({f'name{i}': name[i] for i in range(nums_len)})
    envdict['item'] = item
    envdict['unit_'] = unit_

    # print 포함
    return gen.build(
        # body is the background of problem settings
        body=body,
        question='이 {item}의 총 몇 {unit_}' + ques_trailing,
        equation=gen.EqnRef('extract_from_nums', nums_k, 4),
        env=envdict)


@gen.problems.register
def prob4_01_03sum(selector, tokenpool, clskey):
    '''
    - 은지, 태형, 유나, ..은 각각 주스를 43l, 92l, 71l, 64l 마셨습니다. 가장 많은 주수를 마신사람의 개수에서 가장 적게 주스를 마신 사람의 l를 뺀 수는?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    item = selector.get(clskey.drink)
    nums_len = random.randint(2, 5)
    names = [selector.get(clskey.name) for _ in range(0, nums_len)]

    ## name overlab checkunit
    while len(set(names)) != len(names):
        names = [selector.get(clskey.name) for _ in range(0, nums_len)]

    nums_k = tokenpool.sample(range(0, 100), nums_len)

    # syntactic randomize
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    body = ' '.join([
        ' , '.join('{' + 'name{}'.format(x) + '}' for x in range(nums_len)),
        '은 각각 {item}{#를}',
        '{nums}',
        '를 마셨습니다.'
    ])

    envdict = {'nums': nums_k}
    envdict.update({f'name{i}': names[i] for i in range(nums_len)})
    envdict['item'] = item

    # print 포함
    return gen.build(
        # body is the background of problem settings
        body=body,
        question='사람들이 마신 전체 {item}{#은} 몇 l' + ques_trailing,
        equation=gen.EqnRef('extract_from_nums', nums_k, 4),
        env=envdict)


@gen.problems.register
def prob4_01_04sum(selector, tokenpool, clskey):
    '''
    - 상자에 사탕이 각각 43, 92, 71, 64 개 들어있습니다. 사탕이 가장 많이 담긴 상자의 사탕수에서 ?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    container = selector.get(clskey.container)
    item = selector.get(clskey.flower)
    nums_len = random.randint(2, 5)
    unit = item.of('unit')

    nums_k = tokenpool.sample(range(0, 100), nums_len)

    # syntactic randomize
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    body = ' '.join(['{container}에 {item}이 각각 ',
                     '{nums}',
                     '있습니다.'
                     ])

    envdict = {'nums': nums_k, 'ques_trailing': ques_trailing, 'container': container, 'unit': unit, 'item': item}

    # print 포함
    return gen.build(
        # body is the background of problem settings
        body=body,
        question='{item}의 수를 모두 더하면 몇{ques_trailing}',
        equation=gen.EqnRef('extract_from_nums', nums_k, 4),
        env=envdict)


@gen.problems.register
def prob4_01_05sum(selector, tokenpool, clskey):
    '''
    - 외숙모의 나이는. 외삼촌의 나이는, ... 가장 나이가 많은 사람의 나이에서 나이가 적은 사람의 나이를 뺀 수는?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    l = random.randint(2, 5)
    name = [selector.get(clskey.female_family_relation) for _ in range(0, l)]
    nums = [random.randint(0, 100) for _ in range(0, l)]

    ## name overlab check
    while len(set(name)) != len(name):
        name = [selector.get(clskey.female_family_relation) for _ in range(0, l)]
    nums_k = list(map(tokenpool.new, nums))

    # syntactic randomize
    ques_trailing = random.choice([' 얼마입니까?', ' 얼마인지 구하시오.'])
    cent_trailing = random.choice(['사람', '가족'])

    body = ' '.join(['유나의',
                     ', '.join('{' + 'name{}'.format(x) + '}' +
                               '의 나이는 {' + 'nums{}'.format(x) + '}살' for x in range(l)),
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
        question='{cent_trailing}의 나이를 모두 더하면 {ques_trailing}?',
        equation=gen.EqnRef('extract_from_each_num', 4, *nums_k),
        env=envdict)


@gen.problems.register
def prob4_01_06sum(selector, tokenpool, clskey):
    '''
    - clskey.race에서 4명의 학생은 각각 ..점..점..점을 얻었습니다. 1등의 점수에서 꼴찌의 점수를 뺀 값은?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    l = random.randint(2, 5)
    race = selector.get(clskey.race)

    nums_k = tokenpool.sample(range(0, 100), l)

    # syntactic randomize
    ques_trailing = random.choice(['몇 점입니까?', '몇 점인지 구하시오.'])

    body = ''.join(['{race}에서 {l}명의 학생은 각각 ',
                    '{nums}',
                    '점을 얻었습니다.'
                    ])

    envdict = {f'nums': nums_k, 'race': race, 'l': l, 'ques_trailing': ques_trailing}

    # print 포함
    return gen.build(
        # body is the background of problem settings
        body=body,
        question='점수는 총 {ques_trailing}',
        equation=gen.EqnRef('extract_from_nums', nums_k, 4),
        env=envdict)


@gen.problems.register
def prob4_01_01mam(selector, tokenpool, clskey):
    '''
    - 43, 92, 71, 64가 있습니다. 그중에서 가장 큰 수에서 가장 작은 수를 뺀 값은 얼마입니까?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    nums_len = random.randint(2, 5)
    nums_k = tokenpool.sample(range(0, 100), nums_len)

    envdict = {f'nums': nums_k}

    # syntactic randomize
    body_trailing = random.choice(['있습니다.', '있다.', '나열되어 있다.', '나열되어 있습니다.'])
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    envdict['ques_trailing'] = ques_trailing

    # print 포함
    return gen.build(
        # body is the background of problem settings
        body='{nums}가 ' + body_trailing,
        question='그 중에서 가장 큰 수와 가장 작은 수를 더한 값은 얼마{ques_trailing}',
        equation=gen.EqnRef('extract_from_nums', nums_k, 5),
        env=envdict)


@gen.problems.register
def prob4_01_02mam(selector, tokenpool, clskey):
    '''
    - 은지, 태형, 유나, ..은 각각 사과 43, 92, 71, 64를 가지고 있습니다. 가장 많은 사과를 가진사람의 개수에서 가장 작은 사과를 가진 사름의 개수를 뺀 수는?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    item = selector.get(clskey.fruit)
    nums_len = random.randint(2, 5)
    nums_k = tokenpool.sample(range(0, 100), nums_len)

    name = [selector.get(clskey.name) for _ in range(0, nums_len)]
    unit_ = item.of('unit')

    ## name overlab checkunit
    while len(set(name)) != len(name):
        name = [selector.get(clskey.name) for _ in range(0, nums_len)]

    # syntactic randomize
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    body = ' '.join([
        ' , '.join('{' + 'name{}'.format(x) + '}' for x in range(nums_len)),
        '은 각각 {item}{#를}',
        '{nums}',
        '{unit_}{#를} 가지고 있습니다.'
    ])

    envdict = {f'nums': nums_k}
    envdict.update({f'name{i}': name[i] for i in range(nums_len)})
    envdict['item'] = item
    envdict['unit_'] = unit_

    # print 포함
    return gen.build(
        # body is the background of problem settings
        body=body,
        question='많은 {item}{#를} 가진 사람의 개수와 가장 작은 {item}{#를} 가진 사람의 개수를 더한 수는 몇 {unit_}' + ques_trailing,
        equation=gen.EqnRef('extract_from_nums', nums_k, 5),
        env=envdict)


@gen.problems.register
def prob4_01_03mam(selector, tokenpool, clskey):
    '''
    - 은지, 태형, 유나, ..은 각각 주스를 43l, 92l, 71l, 64l 마셨습니다. 가장 많은 주수를 마신사람의 개수에서 가장 적게 주스를 마신 사람의 l를 뺀 수는?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    item = selector.get(clskey.drink)
    nums_len = random.randint(2, 5)
    names = [selector.get(clskey.name) for _ in range(0, nums_len)]

    ## name overlab checkunit
    while len(set(names)) != len(names):
        names = [selector.get(clskey.name) for _ in range(0, nums_len)]

    nums_k = tokenpool.sample(range(0, 100), nums_len)

    # syntactic randomize
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    body = ' '.join([
        ' , '.join('{' + 'name{}'.format(x) + '}' for x in range(nums_len)),
        '은 각각 {item}{#를}',
        '{nums}',
        '를 마셨습니다.'
    ])

    envdict = {'nums': nums_k}
    envdict.update({f'name{i}': names[i] for i in range(nums_len)})
    envdict['item'] = item

    # print 포함
    return gen.build(
        # body is the background of problem settings
        body=body,
        question='가장 많은 {item}{#를} 마신 사람의 l에서 가장 적은 {item}{#를} 마신 사람의 l를 더하면 몇 l' + ques_trailing,
        equation=gen.EqnRef('extract_from_nums', nums_k, 5),
        env=envdict)


@gen.problems.register
def prob4_01_04mam(selector, tokenpool, clskey):
    '''
    - 상자에 사탕이 각각 43, 92, 71, 64 개 들어있습니다. 사탕이 가장 많이 담긴 상자의 사탕수에서 ?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    container = selector.get(clskey.container)
    item = selector.get(clskey.flower)
    nums_len = random.randint(2, 5)
    unit = item.of('unit')

    nums_k = tokenpool.sample(range(0, 100), nums_len)

    # syntactic randomize
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    body = ' '.join(['{container}에 {item}이 각각 ',
                     '{nums}',
                     '있습니다.'
                     ])

    envdict = {'nums': nums_k, 'ques_trailing': ques_trailing, 'container': container, 'unit': unit, 'item': item}

    # print 포함
    return gen.build(
        # body is the background of problem settings
        body=body,
        question='{item}{#이} 가장 많이 담긴 {item} 수에서 {item}{#이} 가장 적게 담긴 수를 더한 수는 몇{ques_trailing}',
        equation=gen.EqnRef('extract_from_nums', nums_k, 5),
        env=envdict)


@gen.problems.register
def prob4_01_05mam(selector, tokenpool, clskey):
    '''
    - 외숙모의 나이는. 외삼촌의 나이는, ... 가장 나이가 많은 사람의 나이에서 나이가 적은 사람의 나이를 뺀 수는?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    l = random.randint(2, 5)
    name = [selector.get(clskey.female_family_relation) for _ in range(0, l)]
    nums = [random.randint(0, 100) for _ in range(0, l)]

    ## name overlab check
    while len(set(name)) != len(name):
        name = [selector.get(clskey.female_family_relation) for _ in range(0, l)]
    nums_k = list(map(tokenpool.new, nums))

    # syntactic randomize
    ques_trailing = random.choice([' 수는 무엇입니까?', ' 수를 구하시오.'])
    cent_trailing = random.choice(['사람', '가족'])

    body = ' '.join(['유나의',
                     ', '.join('{' + 'name{}'.format(x) + '}' +
                               '의 나이는 {' + 'nums{}'.format(x) + '}살' for x in range(l)),
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
        question='가장 나이가 많은 {cent_trailing}의 나이에서 가장 나이가 적은 {cent_trailing}의 나이를 더한{ques_trailing}?',
        equation=gen.EqnRef('extract_from_each_num', 5, *nums_k),
        env=envdict)


@gen.problems.register
def prob4_01_06mam(selector, tokenpool, clskey):
    '''
    - clskey.race에서 4명의 학생은 각각 ..점..점..점을 얻었습니다. 1등의 점수에서 꼴찌의 점수를 뺀 값은?
    '''
    # this may be a meta-token (e.g., #1), referring a number.
    l = random.randint(2, 5)
    race = selector.get(clskey.race)

    nums_k = tokenpool.sample(range(0, 100), l)
    end = random.choice(['꼴찌의', '꼴등의', '최하위', '마지막 등수의'])

    # syntactic randomize
    ques_trailing = random.choice(['은 무엇입니까?', '을 구하시오.'])

    body = ''.join(['{race}에서 {l}명의 학생은 각각 ',
                    '{nums}',
                    '점을 얻었습니다.'
                    ])

    envdict = {f'nums': nums_k, 'race': race, 'l': l, 'ques_trailing': ques_trailing, 'end': end}

    # print 포함
    return gen.build(
        # body is the background of problem settings
        body=body,
        question='1등의 점수에서 {end}의 점수를 더한 값{ques_trailing}',
        equation=gen.EqnRef('extract_from_nums', nums_k, 5),
        env=envdict)
