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
@gen.equations.register('sum')
def eqn00(*args):
    # return variable is ALWAYS [ans].
    return 'ans = sum([{}])'.format(', '.join(map(str, args)))


@gen.problems.register
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
                '{container}에 {item} {count1}{#이} 있{sent_trailing}',
                '{name}{#이?}가 {container}에서 {item} {count2.to_korunit()}{#을} 꺼냈{sent_trailing}'
                # note that we did not specify a unit.
            ]),
            # question is the main sentence of the problem
            question='{container}에 있는 {item}{#은} {total}몇 {unit}{ques_trailing}',
            equation=gen.EqnRef('sum', count1_k, -count2_k),

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
    return 'ans = {} // {}'.format(result, factor)


@gen.problems.register
def prob02(selector, tokenpool, clskey):
    a = random.randint(2, 19)
    b = random.randint(2, 19) * a

    n1 = tokenpool.new(a)
    n2 = tokenpool.new(b)

    return gen.build(
            body='어떤 수에 {n1}을 곱하면 {n2}가 나온다.',
            question='어떤 수를 구하시오.',
            equation=gen.EqnRef('eqn2', n1, n2),

            env=gen.fnmap(
                n1=n1,
                n2=n2
            ))

# @gen.problems.register
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


@gen.equations.register('average')
def eqn3(*args):
    return 'ans = sum([{}]) / {}'.format(', '.join(map(str, args)), str(len(args)))


# You must prepend @register for each function!
@gen.problems.register
def showcase(sel, pl, clskey):
    # get a real number from [0, 2]
    # this will round the numbers to the 1/100's digit.
    num1 = randreal(0, 2)
    num2 = randreal(0, 2, ndigits=3)  # 10^-3's digit.

    num1_k = pl.new(num1)
    num2_k = pl.new(num2)

    return gen.build(
            body='',
            question='{n1}{#과} {n2}의 평균은 얼마입니까?',
            equation=gen.EqnRef('average', num1_k, num2_k),
            # if answer is not stable (due to floating point arithmetic)
            # specify a stable answer.
            answer=round((num1 + num2) / 2, ndigits=2),

            env=gen.fnmap(
                n1=num1_k,
                n2=num2_k
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
    #clskey.jshape  = 'entity.jshape'
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

    # clskey.big_or_small = 'prop.big_or_small'
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

    # dictionary.add_token(clskey.big_or_small, gen.DictItem('큰'))
    # dictionary.add_token(clskey.big_or_small, gen.DictItem('작은'))


    dictionary.add_token(clskey.ord_rel, gen.DictItem('크', reverse='작', var='큽니', reverse_var='작습니', adv='큰', reverse_adv='작은'))
    dictionary.add_token(clskey.ord_rel, gen.DictItem('무겁', reverse='가볍', var='무겁습니', reverse_var='가볍습니', adv='무거운', reverse_adv='가벼운'))
    dictionary.add_token(clskey.ord_rel, gen.DictItem('빠르', reverse='느리', var='빠릅니', reverse_var='느립니', adv='빠른', reverse_adv='느린'))
    # dictionary.add_token(clskey.ord_rel, gen.DictItem('높', reverse='낮', var='높습니', reverse_var='낮습니', adv='높은', reverse_adv='낮은'))
    # dictionary.add_token(clskey.ord_rel, gen.DictItem('많', reverse='적', var='많습니', reverse_var='적습니', adv='많은', reverse_adv='적은'))
    # dictionary.add_token(clskey.ord_rel, gen.DictItem('길', reverse='짧', var='깁니', reverse_var='짧습니', adv='긴', reverse_adv='짧은'))

    # dictionary.add_token(clskey.ord_rel, gen.DictItem('오른', reverse=['왼']))
    # dictionary.add_token(clskey.ord_rel, gen.DictItem('앞', reverse=['뒤']))

    ######## ADD MAPPINGS ########
    ######## ADJUST HIERARCHY ########
    ######## ADD TOKENS ########

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
                print (ret)
                i += 1
