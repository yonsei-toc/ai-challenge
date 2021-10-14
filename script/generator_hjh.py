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


#names of equations t.b.d --> c{#n}p{#n}

@gen.problems.register
def c5p1(sel, tokenpool, clskey):
    choose = random.randint(0,2)
    name1 = sel.get(clskey.alpha)
    name2 = sel.get(clskey.alpha)

    # 숫자는 한자리 숫자여야함
    num1 = random.randint(1,9)
    num2 = random.randint(1,9)
    num3 = random.randint(1,9)
    num4 = random.randint(1,9)

    name1_token = tokenpool.new(name1)
    name2_token = tokenpool.new(name2)

    num1_token = tokenpool.new(num1)

# body0 -> name1, name2, num1, num2, num3, num4
    if choose==0:
        num2_token = tokenpool.new(num2)
        num3_token = tokenpool.new(num3)
        num4_token = tokenpool.new(num4)
        body0 = ' '.join([
            '{name1}, {name2}{#는} 한 자리 수입니다.',
            '{name1}{#는} {num1}보다 {num2} 큰 수이고, {name2}보다 {num3} 작은 수는 {num4}입니다.'
            ])
        question0 = '{name1}{#와} {name2}의 합을 구하시오.'
        equation0 = gen.EqnRef('sum',num1_token, num2_token, num3_token, num4_token)
        return gen.build(
            body = body0, 
            question = question0,
            equation = equation0, 

            env = gen.fnmap(
                name1=name1_token,
                name2=name2_token,
                num1=num1_token,
                num2=num2_token,
                num3=num3_token,
                num4=num4_token
        ))
    name_aug = sel.get(clskey.name)
# body1 -> name1, num1, num2, num3
    if choose==1:
        num2_token = tokenpool.new(num2)
        num3_token = tokenpool.new(num3)
        body1 = '{name_aug}{#는} {name1}에 {num1}{#을} 더해 {num2}{#을} 얻었습니다.'
        question1 = '{name1}에 {num3}{#을} 더한 결과를 구하시오.'
        equation1 = gen.EqnRef('sum',num1_token, num2_token, num3_token)
        return gen.build(
            body = body1, 
            question = question1,
            equation = equation1, 

            env = gen.fnmap(
                name1=name1_token,
                name2=name2_token,
                num1=num1_token,
                num2=num2_token,
                num3=num3_token,
                name_aug=name_aug
        ))

# body2 -> name1, name2, num1
    if choose==2:
        question2 = '덧셈식 \'{name1}+{name2}={num1}\'에서 {name1}{#와} {name2}의 합을 구하시오.'
        equation2 = gen.EqnRef('sum',num1_token)

        return gen.build(
            body='',
            question = question2,
            equation = equation2, 

            env = gen.fnmap(
                name1=name1_token,
                name2=name2_token,
                num1=num1_token
        ))

@gen.equations.register('c5p2')
def eq_c5p2(n1,n2,n3,n4):
    # 상수 사용
    # range -> 한 자리 수 범위
    # cand이 여러 후보군이 될 수 있음에 따라 indexing
    return  f'''cand = [(var1,var2) for var1 in range(10) for var2 in range(10) if {n1}*10 + var1 - var2*10 - {n2} == {n3}*10 + {n4} and var1!=var2]
(var1,var2)=cand[0]
ans = sum([var1,var2])'''

@gen.problems.register
def c5p2(sel,tokenpool,clskey):
    choose = random.randint(0,2)
    #서로 다른 두 수 A, B가 있습니다. 두 자리 수끼리의 뺄셈식 8A-B2=45에서 A와 B의 합을 구하시오.

    name1 = sel.get(clskey.alpha)
    name2 = sel.get(clskey.alpha)

    # 만약 A,B 혹은 등식의 조건 만족 안하면 다시 뽑아야겠지?
    while(True):
        cand = ''
        num1 = random.randint(1,9)
        num2 = random.randint(0,9)
        num3 = random.randint(1,9)
        num4 = random.randint(0,9)

        equation = f'[(var1,var2) for var1 in range(10) for var2 in range(10) if \
        {num1}*10 + var1 - var2*10 - {num2} == {num3}*10 + {num4} and var1!=var2]'
        cand = eval(equation)
        if len(cand)!=0: break

    name1_token = tokenpool.new(name1)
    name2_token = tokenpool.new(name2)

    num1_token = tokenpool.new(num1)
    num2_token = tokenpool.new(num2)
    num3_token = tokenpool.new(num3)
    num4_token = tokenpool.new(num4)

# body0 -> name1, name2, num1, num2, num3, num4
    if choose==0:
        body0 = '서로 다른 두 수 {name1}, {name2}{#이} 있습니다.',
        question0 = '두 자리 수끼리의 뻴셈식 {num1}{name1}-{name2}{num2}={num3}{num4}에서 {name1}{#와} {name2}의 합을 구하시오.',
        equation0 = gen.EqnRef('c5p2',num1_token,num2_token,num3_token,num4_token)
        return gen.build(    
            body = body0,
            question = question0,
            equation = equation0,

            env = gen.fnmap(
                name1=name1_token,
                name2=name2_token,
                num1=num1_token,
                num2=num2_token,
                num3=num3_token,
                num4=num4_token
                ))

    name_aug = sel.get(clskey.name)
    school = sel.get(clskey.school)
# school 대신에 place로 해도 무방하려나??
# body1 -> name1, name2, name_aug, num1, num2, num3, num4
    if choose==1:
        body1 = ' '.join([
            '{name_aug}{#는} 서로 다른 숫자가 적힌 카드 {name1}{#와} {name2}{#가} 있습니다.',
            '{name_aug}{#가} 숫자 카드를 통해 식 \'{num1}{name1}-{name2}{num2}={num3}{num4}\'{#을} 만들었습니다.'
        ])
        question1 = '{name1}{#와} {name2}의 합을 구하십시오.'
        equation1 = gen.EqnRef('c5p2',num1_token,num2_token,num3_token,num4_token)
        return gen.build(    
            body = body1,
            question = question1,
            equation = equation1,

            env = gen.fnmap(
                name1=name1_token,
                name2=name2_token,
                num1=num1_token,
                num2=num2_token,
                num3=num3_token,
                num4=num4_token,
                name_aug=name_aug
                ))

# body2 -> name1, name2, name_aug, num1, num2, num3, num4, school
# 만약 body에 ' 나오는 경우 어쩌지? 바깥을 "로 감싸나
    if choose==2:
        body2 = ' '.join([
            '오늘 {name_aug}의 {school}에서 수학 시간에 숫자 카드를 가지고 식을 만드는 놀이를 했습니다.',
            '{name_aug}는 연산식 {num1}{name1}-{name2}{num2}={num3}{num4}{#을} 만족하는 {name1}{#와} {name2}{#을} 뽑으려 합니다.'
        ])
        question2 = '{name1}의 값과 {name2}의 값이 다를 때, 둘의 합을 구하시오.'
        equation2 = gen.EqnRef('c5p2',num1_token,num2_token,num3_token,num4_token)
        
        
        return gen.build(    
            body = body2,
            question = question2,
            equation = equation2,

            env = gen.fnmap(
                name1=name1_token,
                name2=name2_token,
                num1=num1_token,
                num2=num2_token,
                num3=num3_token,
                num4=num4_token,
                name_aug=name_aug,
                school=school
                ))

@gen.equations.register('c5p3')
def eq_c5p3(n1,n2,n3,n4,n5):
    # 상수 사용
    # range -> 한 자리 수 범위
    # cand이 여러 후보군이 될 수 있음에 따라 indexing
    # 자릿수에 해당 -> 100,10 등을 곱함
    return f'''cand = [(var1,var2,var3,var4) for var1 in range(10) for var2 in range(10) for var3 in range(10) for var4 in range(10) if {n1}*100 + var1*10 + {n2} + var2*100 + {n3}*10 + var3 == var4*100 + {n4}*10 + {n5} and var1!=var2!=var3!=var4]
(var1,var2,var3,var4)=cand[0]
ans = var4'''
@gen.problems.register
def c5p3(sel,tokenpool,clskey):
    choose = random.randint(0,2)
    #서로 다른 네 수 A, B, C, D가 있습니다. 세 자리 수끼리의 덧셈식 7A4+B6C=D29에서 D를 구하시오.
    # 문제 특성상 29 이런걸 나누는게 맞을까? -> 현재는 29 -> #3#4

    name1 = sel.get(clskey.alpha)
    name2 = sel.get(clskey.alpha)
    name3 = sel.get(clskey.alpha)
    name4 = sel.get(clskey.alpha)
    # 아 매번 확인해야하냐고~~~~
    while(True):
        cand=''
        num1 = random.randint(1,9)
        num2 = random.randint(0,9)
        num3 = random.randint(0,9)
        num4 = random.randint(0,9)
        num5 = random.randint(num2,9)

        equation = f'[(var1,var2,var3,var4) \
        for var1 in range(10) for var2 in range(10) for var3 in range(10) for var4 in range(10) if \
        {num1}*100 + var1*10 + {num2} + var2*100 + {num3}*10 + var3 == var4*100 + {num4}*10 + {num5}]'
        cand=eval(equation)
        if len(cand)==0: continue
        if len(set(cand[0])) ==4: break

    name1_token = tokenpool.new(name1)
    name2_token = tokenpool.new(name2)
    name3_token = tokenpool.new(name3)
    name4_token = tokenpool.new(name4)

    num1_token = tokenpool.new(num1)
    num2_token = tokenpool.new(num2)
    num3_token = tokenpool.new(num3)
    num4_token = tokenpool.new(num4)
    num5_token = tokenpool.new(num5)

# body0 -> name1, name2, name3, name4, num1, num2, num3, num4, num5
    if choose ==0:
        body0 = '서로 다른 네 수 {name1}, {name2}, {name3}, {name4}{#가} 있습니다.'
        question0 = '세 자리 수끼리의 덧셈식 {num1}{name1}{num2}+{name2}{num3}{name3}={name4}{num4}{num5}에서 {name4}{#를} 구하시오.'
        equation0 = gen.EqnRef('c5p3', num1_token, num2_token, num3_token, num4_token, num5_token)
        return gen.build(
            body = body0,
            question = question0,
            equation = equation0,

            env = gen.fnmap(
                name1 = name1_token,
                name2 = name2_token,
                name3 = name3_token,
                name4 = name4_token,
                num1 = num1_token,
                num2 = num2_token,
                num3 = num3_token,
                num4 = num4_token,
                num5 = num5_token
        ))

    name_aug = sel.get(clskey.name)
    school = sel.get(clskey.school)
# body1 -> name1, name2, name3, name4, name_aug, school, num1, num2, num3, num4, num5
    if choose==1:
        body1 = ' '.join([
            '{name_aug}{#는} 서로 다른 숫자를 가진 카드 {name1}, {name2}, {name3}, {name4}{#가} 있습니다.',
            '{name_aug}{#가} 숫자 카드를 통해 식 \'{num1}{name1}{num2}+{name2}{num3}{name3}={name4}{num4}{num5}\'{#을} 만들었습니다.'
        ])
        question1 = '{name4}{#을} 구하십시오.'
        equation1 = gen.EqnRef('c5p3', num1_token, num2_token, num3_token, num4_token, num5_token)
        return gen.build(
            body = body1,
            question = question1,
            equation = equation1,

            env = gen.fnmap(
                name1 = name1_token,
                name2 = name2_token,
                name3 = name3_token,
                name4 = name4_token,
                num1 = num1_token,
                num2 = num2_token,
                num3 = num3_token,
                num4 = num4_token,
                num5 = num5_token,
                name_aug=name_aug,
                school = school
        ))

    timeline = random.choice(['지난주에', '그저께', '어제', '오늘', '이번주에'])
# body2 -> name1, name2, name3, name4, name_aug, school, timeline, num1, num2, num3, num4, num5
    body2 = ' '.join([
        '{timeline} {name_aug}는 {school}에서 세자리 수의 덧셈에 대해 배웠습니다.',
        '{name_aug}는집에서 복습을 하며  {num1}{name1}-{name2}{num2}={num3}{num4}{#을} 만족하는 {name1}, {name2}, {name3}, {name4}{#을} 뽑으려 합니다.'
    ])
    question2 = '네 숫자의 값이 모두 다를 때, {name4}{#을} 구하시오.'
    equation2 = gen.EqnRef('c5p3', num1_token, num2_token, num3_token, num4_token, num5_token)


    return gen.build(
        body = body2,
        question = question2,
        equation = equation2,

        env = gen.fnmap(
            name1 = name1_token,
            name2 = name2_token,
            name3 = name3_token,
            name4 = name4_token,
            num1 = num1_token,
            num2 = num2_token,
            num3 = num3_token,
            num4 = num4_token,
            num5 = num5_token,
            name_aug = name_aug,
            school = school,
            timeline = timeline
    ))


@gen.equations.register('c5p4')
def eq_c5p4(divisor,quotient,remainder):
    return f'''{remainder} = max(range({divisor}))
ans = {divisor}*{quotient}+{remainder}'''
@gen.problems.register
def c5p4(sel,tokenpool,clskey):
    choose = random.randint(0,2)
    #서로 다른 두 자연수 A, B가 있습니다. A를 17로 나누면 몫은 25이고 나머지는 B가 됩니다. 나머지 B가 가장 큰 수일 때 A를 구하시오.

    name1 = sel.get(clskey.alpha)
    name2 = sel.get(clskey.alpha)

    # 숫자 제한 조건 x
    num1 = random.randint(1,1000)
    num2 = random.randint(1,1000)

    name1_token = tokenpool.new(name1)
    name2_token = tokenpool.new(name2)
    
    num1_token = tokenpool.new(num1)
    num2_token = tokenpool.new(num2)

# body0 -> name1, name2, num1, num2
    if choose==0:
        body0 = ' '.join([
            '서로 다른 두 자연수 {name1}, {name2}{#가} 있습니다.',
            '{name1}{#를} {num1}{#으?}로 나누면 몫은 {num2}{#이?}고 나머지는 {name2}{#가} 됩니다.'
        ])
        question0 = '나머지 {name2}가 가장 큰 수일 때 {name1}{#를} 구하시오.'
        equation0 = gen.EqnRef('c5p4',num1_token,num2_token,name2_token)
        return gen.build(
            body = body0,
            question = question0,
            equation = equation0,

            env = gen.fnmap(
                name1 = name1_token,
                name2 = name2_token,
                num1 = num1_token,
                num2 = num2_token,
            ))

# body1 -> name1, name2, num1, num2
    if choose==1:
        body1 = ' '.join([
            '자연수인 {name1}{#와} {name2}{#가} 있습니다.',
            '{name1}{#을} {num1}{#으?}로 나누려 합니다.'
        ])
        question1 = '몫이 {num2}일 때, 가능한 {name1} 중에서 나머지인 {name2}가 최대일 때의 값을 구하시오.'
        equation1 = gen.EqnRef('c5p4',num1_token,num2_token,name2_token)
        return gen.build(
            body = body1,
            question = question1,
            equation = equation1,

            env = gen.fnmap(
                name1 = name1_token,
                name2 = name2_token,
                num1 = num1_token,
                num2 = num2_token,
            ))

    name_aug = sel.get(clskey.name)
# body2 -> name1, name2, name_aug, num1, num2
    
    if choose==2:
        body2 = ' '.join([
            '{name_aug}{#는} 서로 다른 숫자 {name1}{#와} {name2}{#가} 적힌 카드를 갖고 있습니다.',
            '{name1}{#을} {num1}{#으?}로 나누었을 때, 몫이 {num2}{#이?}고 나머지가 {name2}가 되도록 합니다.'
        ])
        question2 = '나머지가 가장 클 때, {name1}{#은} 얼마일까요?'
        equation2 = gen.EqnRef('c5p4',num1_token,num2_token,name2_token)


        return gen.build(
            body = body2,
            question = question2,
            equation = equation2,

            env = gen.fnmap(
                name1 = name1_token,
                name2 = name2_token,
                num1 = num1_token,
                num2 = num2_token,
                name_aug = name_aug
            ))

@gen.equations.register('c5p5')
def eq_c5p5(n1,n4,n5,n6):
    # 상수 사용
    # 반올림 확인 위해 비교 -> 1000 곱함
    # 반올림 조건 확인 숫자 -> 5
    return f'''if {n4} == {n1}*1000:
    ans = len([var for var in range({n5},{n6}) if var < 5])
else:
    ans = len([var for var in range({n5},{n6}) if var >= 5])'''

@gen.problems.register
def c5p5(sel,tokenpool,clskey):
    choose = random.randint(0,2)
    #네 자리 수 6A42를 백의 자리에서 반올림하면 6000이 됩니다. 0부터 9까지의 숫자 중 A에 쓸 수 있는 숫자는 모두 몇 개입니까?
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])
    
    num1 = random.randint(1,9)
    num2 = random.randint(0,9)
    num3 = random.randint(0,9)
    # round_num = random.choice([1,10,100])
    #  ~~의 자리로 만드는 것은 좀 어려울 것 같으니 일단 보류
    num4 = random.choice([num1,num1+1])*1000
    num5 = random.randint(1,10)
    num6 = random.randint(num5,10)
    
    name1 = sel.get(clskey.alpha)

    num1_token = tokenpool.new(num1)
    num2_token = tokenpool.new(num2)
    num3_token = tokenpool.new(num3)
    num4_token = tokenpool.new(num4)
    num5_token = tokenpool.new(num5)
    num6_token = tokenpool.new(num6)

    name1_token = tokenpool.new(name1)
    name_aug = sel.get(clskey.name)
    school = sel.get(clskey.school)
    place = sel.get(clskey.place)
    timeline = random.choice(['지난주에', '그저께', '어제', '오늘', '이번주에'])

# body0 -> name1, name2, num1, num2, num3, num4, num5, num6, total, ques_trailing
    if choose==0:
        body0 = '네 자리 수 {num1}{name1}{num2}{num3}{#를} 백의 자리에서 반올림하면 {num4}{#이} 됩니다.'
        question0 = '{num5}부터 {num6}까지의 숫자 중 {name1}에 쓸 수 있는 숫자는 {total}몇 개{ques_trailing}'
        equation0 = gen.EqnRef('c5p5', num1_token, num4_token, num5_token, num6_token)
        return gen.build(
            body = body0, 
            question = question0,
            equation = equation0,

            env = gen.fnmap(
                num1 = num1_token,
                num2 = num2_token,
                num3 = num3_token,
                num4 = num4_token,
                num5 = num5_token,
                num6 = num6_token,
                name1 = name1_token,
                total = total,
                ques_trailing = ques_trailing,
            ))

# body1 -> name1, name2, num1, num2, num3, num4, num5, num6, total, ques_trailing
    if choose==1:
        body1 = ' '.join([
            '네 자리 수인 {num1}{name1}{num2}{num3}{#가} 있습니다.',
            '이 수를 백의 자리에서 반올림하면 {num4}{#이} 됩니다.'
        ])
        question1 = '이 때, {name1}이 될 수 있는 숫자는 {num5} 이상 {num6} 이하에서 몇 개일까요?'
        equation1 = gen.EqnRef('c5p5', num1_token, num4_token, num5_token, num6_token)

        return gen.build(
            body = body1, 
            question = question1,
            equation = equation1,

            env = gen.fnmap(
                num1 = num1_token,
                num2 = num2_token,
                num3 = num3_token,
                num4 = num4_token,
                num5 = num5_token,
                num6 = num6_token,
                name1 = name1_token,
                total = total,
                ques_trailing = ques_trailing,
                timeline = timeline,
                place = place
            ))
# body2 -> name1, name2, num1, num2, num3, num4, num5, num6, total, ques_trailing, timeline, place
    if choose == 2:
        body2 = ' '.join([
            '{name_aug}{#는} {timeline} {school}에서 반올림에 대해 배웠습니다. ',
            '{place}에 가는 길에 문득 {name_aug}는 네 자리 수 {num1}{name1}{num2}{num3}{#를} 떠올렸습니다.',
        ])
        question2 = '{num1}{name1}{num2}{num3}{#를} 반올림했을 때 {num4}{#이} 된다면, {num5}에서 {num6}까지의 수 중에 {name1}이 될 수 있는 수는 {total}몇 개{ques_trailing}'
        equation2 = gen.EqnRef('c5p5', num1_token, num4_token, num5_token, num6_token)

        # 반올림되는 숫자 뭔가 지정해줘야 하나? 5>=
        return gen.build(
            body = body2, 
            question = question2,
            equation = equation2,

            env = gen.fnmap(
                num1 = num1_token,
                num2 = num2_token,
                num3 = num3_token,
                num4 = num4_token,
                num5 = num5_token,
                num6 = num6_token,
                name1 = name1_token,
                total = total,
                ques_trailing = ques_trailing,
                name_aug = name_aug,
                school = school,
                timeline = timeline,
                place = place
            ))


@gen.equations.register('c6p5')
def eq_c6p5(n1,n2,n3):
    return f'ans = {n1}*{n2}*{n3}'

@gen.problems.register
def c6p5(sel,tokenpool,clskey):
    choose = random.randint(0,2)
    #12에 어떤 수를 곱해야 하는데 잘못하여 어떤 수를 12로 나누었더니 8이 되었습니다. 바르게 계산한 결과를 구하시오.
    name1 = sel.get(clskey.alpha)
    # 숫자 조건 x
    num1 = random.randint(1,100)
    num2 = random.randint(1,100)
    num3 = random.randint(1,100)

    name1_token = tokenpool.new(name1)
    num1_token = tokenpool.new(num1)
    num2_token = tokenpool.new(num2)
    num3_token = tokenpool.new(num3)

# body0 -> name1, num1, num2, num3
    if choose==0:
        body0 = '{num1}에 {name1}{#를} 곱해야 하는데 잘못하여 {name1}{#를} {num2}{#으?}로 나누었더니 {num3}{#이} 되었습니다.'
        question0 = '바르게 계산한 결과를 구하시오.'
        equation0 = gen.EqnRef('c6p5',num1_token,num2_token,num3_token)
        return gen.build(
            body = body0,
            question = question0,
            equation = equation0,

        env = gen.fnmap(
            name1 = name1_token,
            num1 = num1_token,
            num2 = num2_token,
            num3 = num3_token
        ))

# body1 -> name1, num1, num2, num3
    if choose==1:
        body1 = ' '.join([
            '{num1}에 {name1}{#을} 곱하는 식이 있습니다.',
            '전달 중에 식이 바뀌어 {name1}{#을} {num2}{#으?}로 나누는 식이 되었습니다.',
            '몫은 {num3}이고 나머지는 없습니다.'
        ])
        question1 = '원래 식의 결과는 몇일까요?'
        equation1 = gen.EqnRef('c6p5',num1_token,num2_token,num3_token)
        return gen.build(
            body = body1,
            question = question1,
            equation = equation1,

        env = gen.fnmap(
            name1 = name1_token,
            num1 = num1_token,
            num2 = num2_token,
            num3 = num3_token
        ))

    name_aug1 = sel.get(clskey.name)
    name_aug2 = sel.get(clskey.name)
    school = sel.get(clskey.school)

# body2 -> name1, num1, num2, num3, name_aug1, name_aug2, school
    if choose==2:
        body2 = ' '.join([
            '{school}에 등교한 {name_aug1}{#는} 칠판에 적힌 식을 발견했습니다.',
            '식은 {num1}에 {name1}{#을} 곱하는 식입니다.',
            '식이 마음에 들지 않았던 {name_aug1}{#는} 이 식을 {name1}{#을} {num2}로 나누는 식으로 바꾸었습니다.',
            '이후에 등교한 {name_aug2}는 바뀐 식을 계산해 {num3}{#이?}라는 답을 구했습니다.'
        ])
        question2 = '원래의 식을 바르게 계산한 결과를 구하십시오.'
        equation2 = gen.EqnRef('c6p5',num1_token,num2_token,num3_token)

        return gen.build(
            body = body2,
            question = question2,
            equation = equation2,

        env = gen.fnmap(
            name1 = name1_token,
            num1 = num1_token,
            num2 = num2_token,
            num3 = num3_token,
            name_aug1 = name_aug1,
            name_aug2 = name_aug2,
            school = school
        ))


@gen.equations.register('c7p5')
def eq_c7p5(t1,t2,t3,t4,t5,index):
    return f'''sorted_ts=["{t1}","{t5}","{t2}","{t4}","{t3}"]; ans = sorted_ts[{index}-1]'''
@gen.problems.register
def c7p5(sel,tokenpool,clskey):
    choose = random.randint(0,2)
    #철수, 영수, 영철, 경수, 경환 5명이 있습니다. 철수는 나이가 가장 적습니다. 영수는 경수에게는 동생이고 경환에게는 형입니다.
    #경수는 2년 후에 40살이 되고, 영철이는 올해 40살입니다. 5명 중에서 나이가 2번째로 적은 사람은 누구입니까?
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])
    
    name1 = sel.get(clskey.name)
    name2 = sel.get(clskey.name)
    name3 = sel.get(clskey.name)
    name4 = sel.get(clskey.name)
    name5 = sel.get(clskey.name)
    # 일단 5명 고정 - 인원 달라지면 조건문도 달라짐
    names = [name1,name2,name3,name4,name5]
    name_num = len(names)
    
    num1 = random.randint(1,9)
    num2 = random.randint(num1, 100)
    num3 = random.randint(num2,100)
    num4 = random.randint(1,5)

    name1_token = tokenpool.new(name1)
    name2_token = tokenpool.new(name2)
    name3_token = tokenpool.new(name3)
    name4_token = tokenpool.new(name4)
    name5_token = tokenpool.new(name5)

    num4_token = tokenpool.new(num4)
    timeline = random.choice(['지난주에', '그저께', '어제', '오늘', '이번주에'])
    name_aug = sel.get(clskey.name)

# body0 -> name1, name2, name3, name4, name5, name_num, ques_trailing, num1, num2, num3, num4
    if choose==0:
        body0 = ' '.join([
            '{name1}, {name2}, {name3}, {name4}, {name5} {name_num}명이 있습니다.',
            '{name1}{#는} 나이가 가장 적습니다.',
            '{name1}{#는} {name4}{#이?}에게는 동생이고 {name5}{#이?}에게는 형입니다.',
            '{name3}{#는} {num1}년 후에 {num2}살이 되고, {name2}{#이?}는 올해 {num3}살입니다.'
        ])
        question0 = '{name_num}명 중에서 나이가 {num4}번째로 적은 사람은 누구{ques_trailing}'
        equation0 = gen.EqnRef('c7p5', name1_token,name2_token,name3_token,name4_token,name5_token,num4_token)

        return gen.build(
            body = body0,
            question = question0,
            equation = equation0,

            env = gen.fnmap(
                name1 = name1_token,
                name2 = name2_token,
                name3 = name3_token,
                name4 = name4_token,
                name5 = name5_token,
                num1=num1,
                num2=num2,
                num3=num3,
                num4=num4_token,
                name_num=name_num,
                ques_trailing=ques_trailing
        ))
# body1 -> name1, name2, name3, name4, name5, name_num, ques_trailing, num1, num2, num3, num4, timeline
    if choose==1:
        body1 = '{timeline} {name1}, {name2}, {name3}, {name4}, {name5} {name_num}명이 모였습니다.'
        body1 += f' 오랜만에 모인 {gen.korutil.num2korunit(name_num)} 사람들은 서로의 나이가 기억나지 않아 나이를 물어보았습니다.'
        body1 += ' '.join([
            ' 가장 나이가 적은 사람은 {name1}{#이?}입니다.',
            '{name1}{#은} {name4}{#이?}보다 나이가 적고 {name5}보다 나이가 많습니다.',
            '{name3}{#는} {name5}보다 나이가 많고 {name4}보다 나이가 적습니다.'
        ])
        question1 = '올해 {name2}의 나이가 {num3}살이고 {num1}년 후에 {name3}의 나이가 {num2}살이 될 때, 나이가 {num4}번째로 적은 사람을 구하시오.'
        equation1 = gen.EqnRef('c7p5', name1_token,name2_token,name3_token,name4_token,name5_token,num4_token)

        return gen.build(
            body = body1, 
            question = question1,
            equation = equation1, 

            env = gen.fnmap(
                name1 = name1_token,
                name2 = name2_token,
                name3 = name3_token,
                name4 = name4_token,
                name5 = name5_token,
                num1=num1,
                num2=num2,
                num3=num3,
                num4=num4_token,
                name_num=name_num,
                ques_trailing=ques_trailing,
                timeline=timeline
        ))
# body2 -> name1, name2, name3, name4, name5, name_num, ques_trailing, num1, num2, num3, num4, name_aug
    if choose ==2:
        body2 = '{name_aug}는 {name1}, {name2}, {name3}, {name4}, {name5}의 사람들을 일렬로 세우려 합니다.'
        body2 += ' 고민을 하던 {name_aug}는 나이가 작은 사람부터 순서대로 세우기로 했습니다.'
        body2 += ' '.join([
            ' {name1}{#은} 막내입니다.',
            '{name1}의 앞에는 {name4}{#이?}가 있고 {name5}{#는} {name1}의 앞에 있습니다.',
            '{name3}{#는} {name5}보다 뒤에 있고  {name4}보다 앞에 있습니다.'
        ])

        question2 = '{name2}의 현재 나이는 {num3}의 {num1}년 후의 나이일 때, 나이가 {num4}번째로 적은 사람을 구하시오.'
        equation2 = gen.EqnRef('c7p5', name1_token,name2_token,name3_token,name4_token,name5_token,num4_token)

        return gen.build(
            body = body2,
            question = question2,
            equation = equation2, 

            env = gen.fnmap(
                name1 = name1_token,
                name2 = name2_token,
                name3 = name3_token,
                name4 = name4_token,
                name5 = name5_token,
                num1=num1,
                num2=num2,
                num3=num3,
                num4=num4_token,
                name_num=name_num,
                ques_trailing=ques_trailing,
                name_aug=name_aug
        ))


@gen.equations.register('c8p5')
def eq_c8p5(e1,e2,n1,n2):
    return f'''{e2}={n1} // (2 * ({n2}+1)); {e1} = {e2}*{n2}; ans = {e1}'''
@gen.problems.register
def c8p5(sel,tokenpool,clskey):
    choose = random.randint(0,2)
    #둘레가 24cm인 직사각형이 있습니다. 이 직사각형의 가로 길이가 세로 길이의 2배일 때 가로는 몇 cm입니까?
    while(True):
        num1 = random.randint(10,100)
        num2 = random.randint(1,10)
        if num1 % (2*(num2+1)) == 0: break

    edge1 = sel.get(clskey.edge)
    edge2 = sel.get(clskey.edge)

    num1_token = tokenpool.new(num1)
    num2_token = tokenpool.new(num2)

    unit = sel.get(clskey.length_unit).text
    num1_token.unit = unit

    edge1_token = tokenpool.new(edge1)
    edge2_token = tokenpool.new(edge2)

    name_aug = sel.get(clskey.name)
    figure_type = random.choice(['직','정'])

# body0 -> num1, num2, edge1, edge2, unit, figure_type
    body0 = '둘레가 {num1}인 {figure_type}사각형이 있습니다.'
    question0 = '이 {figure_type}사각형의 {edge1} 길이가 {edge2} 길이의 {num2}배일 때 {edge1}{#는} 몇 {unit}입니까?'
    equation0 = gen.EqnRef('c8p5',edge1_token,edge2_token,num1_token,num2_token)

# body1 -> num1, num2, edge1, edge2, unit, figure_type
    body1 = '{figure_type}사각형의 둘레는 {num1}입니다.'
    question1 = '{edge1}이 {edge2}보다 {num2}배 길 때, {edge1}{#는} 몇 {unit}입니까?'
    equation1 = gen.EqnRef('c8p5',edge1_token,edge2_token,num1_token,num2_token)

# body2 -> num1, num2, edge1, edge2, unit, figure_type, name_aug
    if choose ==2:
        body2 = '{name_aug}{#는} {num1}의 철사를 사용해 {figure_type}사각형을 만들려 합니다.'
        question2 = '{edge1}의 길이가 {edge2} 길이의 {num2}배가 되도록 {figure_type}사각형을 만들 때, {edge1}의 길이를 구하시오.'
        equation2 = gen.EqnRef('c8p5',edge1_token,edge2_token,num1_token,num2_token)
        return gen.build(
            body = body2, 
            question = question2,
            equation = equation2,

            env = gen.fnmap(
                num1 = num1_token,
                num2 = num2_token,
                edge1 = edge1_token,
                edge2 = edge2_token,
                unit = unit,
                figure_type = figure_type,
                name_aug = name_aug
        ))

    if choose == 0:
        return gen.build(
            body = body0,
            question = question0,
            equation = equation0,

            env = gen.fnmap(
                num1 = num1_token,
                num2 = num2_token,
                edge1 = edge1_token,
                edge2 = edge2_token,
                unit = unit,
                figure_type = figure_type
        ))
    if choose ==1:
        return gen.build(
            body = body1,
            question = question1,
            equation = equation1,

            env = gen.fnmap(
                num1 = num1_token,
                num2 = num2_token,
                edge1 = edge1_token,
                edge2 = edge2_token,
                unit = unit,
                figure_type = figure_type
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
