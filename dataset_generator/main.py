import template
from template import register, clskey

import random

# add this!
@register
def prob01(tokens):
    # You may alias functions of a long name
    c = tokens.append_connection

    # Claim entities at first. They will not overlap (even for different keys).
    container = tokens.get(clskey.container)
    item = tokens.get(clskey.item)
    name = tokens.get(clskey.name)

    # randint() may overlab in its values.
    # DO NOT ASSUME that this is an int.
    # this may be a meta-token (e.g., #1), referring a number.
    count1 = tokens.randint(1, 100)
    count2 = tokens.randint(1, count1)

    # syntactic randomize
    total = random.choice(['', '모두 ', '총 ', '다해서 '])
    sent_trailing = random.choice(['습니다.', '다.'])
    ques_trailing = random.choice(['입니까?', '인지 구하시오.', '인가?'])

    # Some tokens have a unit attribute; e.g. item.of("unit")

    # f'...' is formatted string
    # f'{container}' is equivalent to
    # '{container}'.format(container=container) or
    # '{}'.format(container), etc.

    # tokens.append_connection(str, conn) (aliased to c) choose
    # which connection should be used (사과"가", 귤"이", etc.)
    # according to the existence of "jongseong" of the last hangul character.
    # Hint for experts: this uses str() builtin method to get a string.
    body = ' '.join([
        f'{container}에 {item} {count1}{c(item.of("unit"), "이")} 있{sent_trailing}',
        f'{c(name, "이?")}가 {container}에서 {item} {count2.to_korunit()} {c(item.of("unit"), "을")} 꺼냈{sent_trailing}'
        ])

    question = f'{container}에 있는 {c(item, "은")} {total}몇 {item.of("unit")}{ques_trailing}'
    equation = f'ans = sum([{count1}, -{count2}])'

    return template.format(body, question, equation, 'ans')


@register
def prob02(tokens):
    c = tokens.append_connection
    item = tokens.get(clskey.tool_group)
    cvtunit = random.choice(item.of("group_unit"))
    ans = random.randint(1, 10)

    body = ''
    question = f'{item} {ans*cvtunit[1]}{c(item.of("unit"), "은")} 몇 {cvtunit[0]}입니까?'

    # do not recommend this in practice; this is for an illustration purpose..
    variable = ''.join(random.sample('abcdefghijklmnopqrstuvwxyz', k=random.randint(1, 2)))

    equation = f'{variable} = {ans * cvtunit[1]} // {cvtunit[1]}'

    return template.format(body, question, equation, variable)


# You must prepend @template.register for each function!
@register
def showcase(tokens):
    c = tokens.append_connection
    # get a real number from [0, 2]
    # this will round the numbers to the 1/100's digit.
    name = tokens.get(clskey.name)
    num1 = tokens.randreal(0, 2)
    num2 = tokens.randreal(0, 2, ndigits = 3)  # 10^-3's digit.

    #body = f'You have two numbers: {num1} and {num2}.'
    body = ''
    question = f'{c(str(num1), "과")} {num2}의 평균은 얼마입니까?'
    equation = f'ans = sum([{num1}, {num2}]) / 2'

    # if you do not specify a variable (and so do in the equation),
    # templating system will add one.
    # if your equation is not a single sentence, you must specify a variable!

    # due to floating point error, the answer may be unstable.
    # if you do not want that, specify an answer.
    # be careful for rounding, especially for a real number!
    return template.format(body, question, equation, 'ans',
            answer=round((num1.value + num2.value / 2), 2))



if __name__ == '__main__':
    def main():
        dictionary = template.TokenRegistry()

        setup_dictionary(clskey, dictionary)
        for fn in template.templates.fns:
            i = 0
            while i < 3:
                tokens = template.TokenSelector(dictionary)
                ret = fn(tokens)
                if ret is None:
                    continue
                else:
                    i += 1
                    print(ret)

    # You may register additional tokens/classes
    # dict from the 1st doc
    def setup_dictionary(clskey, dictionary):
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

        clskey.color = 'prop.color'


        # setup hierarchy
        # item is a supertype of fruit, etc., etc..
        dictionary.add_hierarchy(clskey.item, clskey.fruit)
        dictionary.add_hierarchy(clskey.item, clskey.tool)
        dictionary.add_hierarchy(clskey.item, clskey.container)
        dictionary.add_hierarchy(clskey.item, clskey.flower)

        dictionary.add_hierarchy(clskey.tool, clskey.tool_group)
        dictionary.add_hierarchy(clskey.tool, clskey.wire)

        # add tokens
        dictionary.add_token(clskey.container, '바구니', unit='개')
        dictionary.add_token(clskey.container, '상자', unit='개')
        # dictionary.add_token(clskey.container, '선반', unit='개')
        dictionary.add_token(clskey.container, '소쿠리', unit='개')
        dictionary.add_token(clskey.container, '그릇', unit='개')
        dictionary.add_token(clskey.container, '접시', unit='개')
        dictionary.add_token(clskey.container, '주머니', unit='개')

        dictionary.add_token(clskey.fruit, '사과', unit='개')
        dictionary.add_token(clskey.fruit, '배', unit='개')
        dictionary.add_token(clskey.fruit, '감', unit='개')
        dictionary.add_token(clskey.fruit, '귤', unit='개')
        dictionary.add_token(clskey.fruit, '포도', unit='송이')
        dictionary.add_token(clskey.fruit, '수박', unit='통')
        dictionary.add_token(clskey.fruit, '키위', unit='개')
        dictionary.add_token(clskey.fruit, '바나나', unit='개')
        # dictionary.add_token(clskey.fruit, '마카다미아', unit='개')

        dictionary.add_token(clskey.animal, '개', unit='마리')
        dictionary.add_token(clskey.animal, '고양이', unit='마리')
        dictionary.add_token(clskey.animal, '소', unit='마리')
        dictionary.add_token(clskey.animal, '말', unit='마리')
        dictionary.add_token(clskey.animal, '오리', unit='마리')
        dictionary.add_token(clskey.animal, '닭', unit='마리')
        dictionary.add_token(clskey.animal, '토끼', unit='마리')
        dictionary.add_token(clskey.animal, '물고기', unit='마리')
        dictionary.add_token(clskey.animal, '고래', unit='마리')
        dictionary.add_token(clskey.animal, '거위', unit='마리')
        dictionary.add_token(clskey.animal, '달팽이', unit='마리')
        dictionary.add_token(clskey.animal, '개구리', unit='마리')
        # ...
        # dictionary.add_token(clskey.animal, '강아지', unit='마리')
        # dictionary.add_token(clskey.animal, '송아지', unit='마리')
        # dictionary.add_token(clskey.animal, '망아지', unit='마리')
        # dictionary.add_token(clskey.animal, '병아리', unit='마리')
        # ...

        dictionary.add_token(clskey.flower, '장미', unit='송이')
        dictionary.add_token(clskey.flower, '백합', unit='송이')
        dictionary.add_token(clskey.flower, '튤립', unit='송이')
        dictionary.add_token(clskey.flower, '카네이션', unit='송이')
        dictionary.add_token(clskey.flower, '국화', unit='송이')
        dictionary.add_token(clskey.flower, '화분', unit='송이')
        dictionary.add_token(clskey.flower, '화단', unit='송이')

        dictionary.add_token(clskey.tool, '책', unit='권')
        dictionary.add_token(clskey.tool, '공책', unit='권')
        dictionary.add_token(clskey.tool, '종이', unit='장')
        dictionary.add_token(clskey.tool, '도화지', unit='장')
        dictionary.add_token(clskey.tool, '색종이', unit='장')
        dictionary.add_token(clskey.tool, '지우개', unit='개')
        dictionary.add_token(clskey.tool, '컵', unit='개')
        dictionary.add_token(clskey.tool, '신발', unit='켤레')
        dictionary.add_token(clskey.tool, '꽃병', unit='병')
        dictionary.add_token(clskey.tool, '배구공', unit='개')
        dictionary.add_token(clskey.tool, '농구공', unit='개')
        dictionary.add_token(clskey.tool, '축구공', unit='개')
        dictionary.add_token(clskey.tool, '탁구공', unit='개')
        dictionary.add_token(clskey.tool, '야구공', unit='개')

        # dictionary.add_token(clskey.tool, '차', unit='대')
        dictionary.add_token(clskey.ride, '자동차', unit='대')
        dictionary.add_token(clskey.ride, '비행기', unit='대')
        dictionary.add_token(clskey.ride, '배', unit='척')
        dictionary.add_token(clskey.ride, '트럭', unit='대')
        dictionary.add_token(clskey.ride, '자전거', unit='대')
        dictionary.add_token(clskey.ride, '오토바이', unit='대')
        dictionary.add_token(clskey.ride, '기차', unit='대')
        dictionary.add_token(clskey.ride, '버스', unit='대')
        dictionary.add_token(clskey.ride, '엘리베이터', unit='대')

        dictionary.add_token(clskey.color, '흰', alias=['하얀'])
        dictionary.add_token(clskey.color, '검은')
        dictionary.add_token(clskey.color, '빨간')
        dictionary.add_token(clskey.color, '노란')
        dictionary.add_token(clskey.color, '초록')
        dictionary.add_token(clskey.color, '파란')
        dictionary.add_token(clskey.color, '보라')

        dictionary.add_token(clskey.tool_group, '연필', unit='자루', group_unit=[('다스', 12)])
        dictionary.add_token(clskey.tool_group, '달걀', unit='알', group_unit=[('판', 30)])

        dictionary.add_token(clskey.wire, '철사', unit='개')
        dictionary.add_token(clskey.wire, '끈', unit='개')
        dictionary.add_token(clskey.wire, '줄', unit='개')

        dictionary.add_token(clskey.place, '서점')
        dictionary.add_token(clskey.place, '마트')
        dictionary.add_token(clskey.place, '문구점')
        dictionary.add_token(clskey.place, '집')
        dictionary.add_token(clskey.place, '학교')
        dictionary.add_token(clskey.place, '수영장')
        dictionary.add_token(clskey.place, '교실')
        dictionary.add_token(clskey.place, '도서관')
        dictionary.add_token(clskey.place, '박물관')
        dictionary.add_token(clskey.place, '운동장')
        dictionary.add_token(clskey.place, '주차장')
        dictionary.add_token(clskey.place, '정류장')
        dictionary.add_token(clskey.place, '아파트')
        dictionary.add_token(clskey.place, '농장')
        dictionary.add_token(clskey.place, '강당')

        dictionary.add_token(clskey.name, '남준')
        dictionary.add_token(clskey.name, '석진')
        dictionary.add_token(clskey.name, '윤기')
        dictionary.add_token(clskey.name, '호석')
        dictionary.add_token(clskey.name, '지민')
        dictionary.add_token(clskey.name, '태형')
        dictionary.add_token(clskey.name, '정국')
        dictionary.add_token(clskey.name, '민영')
        dictionary.add_token(clskey.name, '유정')
        dictionary.add_token(clskey.name, '은지')
        dictionary.add_token(clskey.name, '유나')
        dictionary.add_token(clskey.name, '철수')
        dictionary.add_token(clskey.name, '영희')
        dictionary.add_token(clskey.name, '영수')

        ##########################FOR MAPPINGS, PLEASE ADD HERE##################################

        
        ##########################FOR HIERARCHY, PLEASE ADD HERE#################################

        
        ##########################FOR TOKENS, PLEASE ADD HERE####################################



        # for i in "가나다라마바사아자차카타파하":
        #     dictionary.add_token(clskey.name, '(' + i + ')')

    main()
