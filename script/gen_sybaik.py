from generator import token
from generator.build import fnmap
import generator.exports as gen

import functools
import itertools
import math
import random
import string

from generator.problem import EqnRef

# utils
def randreal(st, ed, *, ndigits = 2):
    if ed is None:
        st, ed = 0, st

    if ndigits is None:
        return random.uniform(st, ed)
    else:
        return round(random.uniform(st, ed), ndigits=ndigits)

###################
##   equations   ##
###################


###################
##   problems    ##
###################

@gen.problems.register
def prob1_1(selector, tokenpool, clskey):
    # entities
    container1 = selector.get(clskey.container)
    item1 = selector.get(clskey.item)
    name1 = selector.get(clskey.name)
    count1 = random.randint(1,100)
    count2 = random.randint(1,100)

    # word for entities
    container1_k = tokenpool.new(container1)
    item1_k = tokenpool.new(item1)
    count1_k = tokenpool.new(count1)
    count2_k = tokenpool.new(count2)

    # unit string
    unit = item1.of('unit')
    count1_k.unit = unit
    count2_k.unit = unit

    # syntactic randomize
    num_add = random.choice(['더 ', '추가로 '])
    total = random.choice(['', '모두 ', '총 ', '다해서 ','전체 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    # build
    return gen.build(
        # body
        body=' '.join([
            '{container1}에 {item1}{#가} {count1}{#이} 있{sent_trailing}',
            '{name1}{#이}가 {container1}에 {item1} {count2}{#를} {num_add}넣었{sent_trailing}'
        ]),
        # question
        question = '{container1} 안에 있는 {item1}{#는} {total}몇 {unit}{ques_trailing}',
        # equation
        equation = gen.EqnRef('eqn_sum', count1_k, count2_k),
        
        # env
        env=gen.fnmap(
            container1 = container1_k,
            item1 = item1_k,
            name1 = name1,
            count1=count1_k,
            count2=count2_k,
            unit=unit,
            num_add=num_add,
            total=total,
            sent_trailing=sent_trailing,
            ques_trailing=ques_trailing
        )
    )

@gen.problems.register
def prob1_2(selector, tokenpool, clskey):
    # entities
    name1 = selector.get(clskey.name)
    relation1 = selector.get(name1.of("family_relation"))
    relation2 = selector.get(name1.of("family_relation"))
    item1 = selector.get(clskey.item)
    count1 = random.randint(1, 100)
    count2 = random.randint(1, 100)
    count3 = random.randint(1, 100)
    
    # word for entities
    relation1_k = tokenpool.new(relation1)
    relation2_k = tokenpool.new(relation2)
    item1_k = tokenpool.new(item1)
    count1_k = tokenpool.new(count1)
    count2_k = tokenpool.new(count2)
    count3_k = tokenpool.new(count3)

    # unit string
    unit = item1.of('unit')
    count1_k.unit = unit
    count2_k.unit = unit
    count3_k.unit = unit

    # syntactic randomize
    init_time = random.choice(['처음에 ','원래 ','기존에 '])
    own_item = random.choice(['가지고 있던 ','소유하는 ','가진 '])
    give_item = random.choice(['주','넘기'])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])
    
    # build
    return gen.build(
        # body
        body = ' '.join([
            '{name1}{#이}가 {own_item}{item1} 중에서 {relation1}에게 {count1}{#를} {give_item}고 {relation2}에게 {count2}{#를} {give_item}었더니 {count3}{#가} 남았{sent_trailing}'
        ]),
        # question
        question = '{init_time}{name1}{#이}가 {own_item}{item1}은 몇 {unit}{ques_trailing}',
        # equation
        equation = gen.EqnRef('eqn_sum', count1_k, count2_k, count3_k),

        # env
        env=gen.fnmap(
            name1 = name1,
            relation1 = relation1_k,
            relation2 = relation2_k,
            item1 = item1_k,
            count1 = count1_k,
            count2 = count2_k,
            count3 = count3_k,
            unit = unit,
            init_time = init_time,
            own_item = own_item,
            give_item = give_item,
            sent_trailing = sent_trailing,
            ques_trailing = ques_trailing
        )
    )

@gen.problems.register
def prob1_3(selector, tokenpool, clskey):
    # entities
    item1 = selector.get(clskey.item)
    item2 = selector.get(clskey.item)
    count1 = random.randint(1,100)
    count2 = random.randint(1,100)
    count3 = count1*2+count2

    # word for entities
    item1_k = tokenpool.new(item1)
    item2_k = tokenpool.new(item2)
    count2_k = tokenpool.new(count2)
    count3_k = tokenpool.new(count3)

    # unit string
    unit = item1.of('unit')
    count2_k.unit = unit
    count3_k.unit = unit

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 ','전체 '])
    sent_trailing = random.choice(['습니다.', '다.'])

    
    return gen.build(
        # body(background of question)
        body = ' '.join([
            '{item1}{#과} {item2}{#이} {total}합해서 {count3}개 있{sent_trailing}'
        ]),

        # what to ask
        question = '{item1}{#이} {item2}보다 {count2}{#가} 더 적다면 {item1}{#은} 몇 {unit} 있습니까?',
        # equation for question
        equation = gen.EqnRef('avg', count3_k, -count2_k),

        # env
        env=gen.fnmap(
            item1 = item1_k,
            item2 = item2_k,
            count2 = count2_k,
            count3 = count3_k,
            unit = unit,
            total = total,
            sent_trailing = sent_trailing
        )
    )

@gen.problems.register
def prob1_4(selector, tokenpool, clskey):
    # entities
    item1 = selector.get(clskey.item)
    count1 = random.randint(1,100)
    count2 = random.randint(1,100)
    count3 = random.randint(1,100)
    count4 = random.randint(1,100)

    # entity reference
    item1_k = tokenpool.new(item1)
    count1_k = tokenpool.new(count1)
    count2_k = tokenpool.new(count2)
    count3_k = tokenpool.new(count3)
    count4_k = tokenpool.new(count4)

    # unit
    unit = item1.of('unit')
    count1_k.unit = unit
    count2_k.unit = unit
    count3_k.unit = unit
    count4_k.unit = unit

    # syntactic randomize
    oops = random.choice(['잘못하여 ','실수하여 ','의도한 것과 다르게 '])
    even_distribution = random.choice(['똑같이 ','동일하게 ','공평하게 '])
    max_distribution = random.choice(['최대한 ','최대 ',''])
    sent_trailing = random.choice(['습니다.', '다.'])

    return gen.build(
        # body(background of question)
        body = ' '.join([
            '{item1}{#를} {count1}명에게 {even_distribution}나누어 주어야 할 것을 {oops}{count2}명에게 {even_distribution}나누어 주었더니 한 사람당 {count3}씩 주고 {count4}{#이} 남았{sent_trailing}'
        ]),
        # what to ask
        question = '이 {item1}{#를} {count1}명에게 {even_distribution}나누어 주면 한 사람당 {max_distribution}몇 {unit}씩 가지게 됩니까?',
        # equation for question
        equation = gen.EqnRef('split_oops_split',count3_k,count2_k,count4_k,count1_k),
        env= gen.fnmap(
            item1 = item1_k,
            count1 = count1_k,
            count2 = count2_k,
            count3 = count3_k,
            count4 = count4_k,
            unit = unit,
            oops = oops,
            even_distribution = even_distribution,
            max_distribution = max_distribution,
            sent_trailing = sent_trailing
        )
    )
    

@gen.problems.register
def prob1_5(selector, tokenpool, clskey):
    
    # entities
    name1 = selector.get(clskey.name)
    group1 = selector.get(random.choice([clskey.gender,clskey.school]))
    item1 = selector.get(clskey.item)
    # entity numbers
    count1 = random.randint(2,100)
    count2 = random.choice(list(filter(lambda n: count1%n==0,range(2,count1+1))))
    count3 = random.randint(1,count2)
    midnum = count1//count2*count3
    if midnum==1:
        count3 = random.randint(2,count2)
        midnum = count1//count2*count3
    count4 = random.choice(list(filter(lambda n: midnum%n==0,range(2,midnum+1))))
    count5 = random.randint(1,count4)

    # entity reference
    group1_k = tokenpool.new(group1)
    item1_k = tokenpool.new(item1)
    count1_k = tokenpool.new(count1)
    count2_k = tokenpool.new(count2)
    count3_k = tokenpool.new(count3)
    count4_k = tokenpool.new(count4)
    count5_k = tokenpool.new(count5)

    # syntactic randomize
    total = random.choice(['', '모든 ', '총 ', '전체 '])
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    return gen.build(
        # body(background of question)
        body = ' '.join([
            '{name1}네 반 {total}학생 수는 {count1}명입니다. 그중에서 {group1} 학생은 전체의 {count3}/{count2}입니다.'
            '{group1} 학생 중에서 {item1}{#을} 가진 학생은 {group1} 학생 전체의 {count5}/{count4}입니다.'
        ]),

        # what to ask
        question = '{name1}네 반에서 {item1}{#을} 가지지 못한 학생 {group1} 학생은 몇 명{ques_trailing}',
        # equation for question
        equation = gen.EqnRef('multi_frac',count1_k,count3_k,count2_k,count5_k,count4_k),

        env=gen.fnmap(
            name1 = name1,
            group1 = group1_k,
            item1 = item1_k,
            count1 = count1_k,
            count2 = count2_k,
            count3 = count3_k,
            count4 = count4_k,
            count5 = count5_k,
            total = total,
            ques_trailing = ques_trailing
        )
    )
    

@gen.problems.register
def prob6_1(selector, tokenpool, clskey):
    # entities
    # entity numbers
    count1 = random.randint(1,100)
    count2 = random.randint(1,100)
    # entity reference
    count1_k = tokenpool.new(count1)
    count2_k = tokenpool.new(count2)

    # syntactic randomize
    sent_trailing = random.choice(['습니다.', '다.'])

    return gen.build(
        # body(background of question)
        body = ' '.join([
            '어떤 수에서 {count1}{#을} 뺐더니 {count2}가 되었{sent_trailing}'
        ]),
        # what to ask
        question = '어떤 수를 구하시오.',
        # equation for question
        equation = gen.EqnRef('eqn_sum',count1,count2),
        env = gen.fnmap(
            count1 = count1_k,
            count2 = count2_k,
            sent_trailing = sent_trailing
        )
    )

@gen.problems.register
def prob7_1(selector, tokenpool, clskey):
    
    # entities
    name1 = selector.get(clskey.name)
    name2 = selector.get(clskey.name)
    name3 = selector.get(clskey.name)
    item1 = selector.get(clskey.item)
    # entity numbers
    count1 = random.randint(1,100)
    count2 = random.randint(1,100)
    count3 = random.randint(1,count1+count2)
    # entity reference
    
    name1_k = tokenpool.new(name1)
    name2_k = tokenpool.new(name2)
    name3_k = tokenpool.new(name3)
    item1_k = tokenpool.new(item1)
    count1_k = tokenpool.new(count1)
    count2_k = tokenpool.new(count2)
    count3_k = tokenpool.new(count3)

    # unit
    unit = item1.of('unit')
    count1_k.unit = unit
    count2_k.unit = unit
    count3_k.unit = unit

    # syntactic randomize
    num_greater = random.choice(['더 많이 ','추가로 ','많이 '])
    num_smaller = random.choice(['더 적게 ','조금 ','부족하게 ','적게 '])
    num_smallest = random.choice(['가장 적게 ','최소로 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    return gen.build(
    # body(background of question)
        body = ' '.join([
            '{name1}{#이?}는 {item1}{#를} {count1} 가지고 있{sent_trailing}'
            '{name2}{#이?}는 {item1}{#를} {name1}{#이?}보다 {count2} {num_greater}가지고 있고 {name3}{#이?}는 {name2}{#이?}보다 {count3} {num_smaller}가지고 있{sent_trailing}'
        ]),
        # what to ask
        question = '{item1}{#를} {num_smallest}가지고 있는 사람은 누구{ques_trailing}',
        # equation for question
        equation = gen.EqnRef('select_small_from_three',name1_k,name2_k,name3_k,count1_k,count2_k,count3_k),
        env=gen.fnmap(
            name1 = name1_k,
            name2 = name2_k,
            name3 = name3_k,
            item1 = item1_k,
            count1= count1_k,
            count2 = count2_k,
            count3 = count3_k,
            unit=unit,
            num_greater=num_greater,
            num_smaller = num_smaller,
            num_smallest = num_smallest,
            sent_trailing = sent_trailing,
            ques_trailing = ques_trailing
        )
    )
    

@gen.problems.register
def prob8_1(selector, tokenpool, clskey):
    # entities
    # entity numbers
    count1 = random.randint(3,100)
    count2 = random.randint(3,100)
    # entity reference
    count1_k = tokenpool.new(count1)
    count2_k = tokenpool.new(count2)
    # syntactic randomize
    return gen.build(
        # body(background of question)
        body = ' '.join([
            f''
        ]),
        # what to ask
        question = '{count1.to_kor()}각형의 변의 개수와 {count2.to_kor()}각형의 변의 개수의 합을 구하시오.',
        # equation for question
        equation = gen.EqnRef('eqn_sum',count1_k,count2_k),
        env = gen.fnmap(
            count1 = count1_k,
            count2 = count2_k
        )
    )



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
