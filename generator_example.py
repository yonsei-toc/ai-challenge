import generator.exports as gen
import random

# it accepts an id. if it is not provided, use the function name.
@gen.equations.register
def eqn(*args):
    # the arguments can be number or string. be careful.
    # return variable is ALWAYS [ans].
    return 'ans = sum([{}])'.format(', '.join(map(str, args)))

@gen.problems.register
def prob01(selector, tokenpool, clskey):
    # Claim entities at first. They will not overlap (even for different keys).
    container = selector.get(clskey.container)
    item = selector.get(clskey.item)
    name = selector.get(clskey.name)
    count1 = random.randint(1, 100)
    count2 = random.randint(1, count1)

    # setup the words to be replaced into tokens
    container_k = tokenpool.new(container)
    item_k = tokenpool.new(item)
    name_k = tokenpool.new(name)

    # the following is equiv. to count1_k = tokenpool.new(random.randint(1, 100))
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
    # i.e., {count2.to_korunit()} uses the same token id as count2.
    # #{이} will be converted "이" or "가" according to the previous character.
    # 은, 는, 이, 가, 을, 를, 와, 과, 이?, 으?
    return gen.build(
            body=' '.join([
                '{container}에 {item} {count1}#{이} 있{sent_trailing}',
                '{name}#{이?}가 {container}에서 {item} {count2.to_korunit()}#{을} 꺼냈{sent_trailing}'
            ]),
            question='{container}에 있는 {item}#{은} {total}몇 {unit}{ques_trailing}',
            equation=gen.EqnRef('eqn', count1_k, -count2_k),

            env = gen.fnmap(
                container=container_k,
                item=item_k,
                name=name_k,
                count1=count1_k,
                count2=count2_k,
                total=total,
                sent_trailing=sent_trailing,
                ques_trailing=ques_trailing,
                unit=unit
            ))

@gen.equations.register
def eqn2(factor, result):
    # the arguments can be number or string. be careful.
    # return variable is ALWAYS [ans].
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

            env = gen.fnmap(
                n1=n1,
                n2=n2
            ))

def build_dictionary(clskey, dictionary):
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
    dictionary.set_child_relation(clskey.item, clskey.fruit)
    dictionary.set_child_relation(clskey.item, clskey.tool)
    dictionary.set_child_relation(clskey.item, clskey.container)
    dictionary.set_child_relation(clskey.item, clskey.flower)

    dictionary.set_child_relation(clskey.tool, clskey.tool_group)
    dictionary.set_child_relation(clskey.tool, clskey.wire)

    # add tokens
    dictionary.add_token(clskey.container, gen.DictItem('바구니', unit='개'))
    dictionary.add_token(clskey.container, gen.DictItem('상자', unit='개'))
    # dictionary.add_token(clskey.container, '선반', unit='개')
    dictionary.add_token(clskey.container, gen.DictItem('소쿠리', unit='개'))
    dictionary.add_token(clskey.container, gen.DictItem('그릇', unit='개'))
    dictionary.add_token(clskey.container, gen.DictItem('접시', unit='개'))
    dictionary.add_token(clskey.container, gen.DictItem('주머니', unit='개'))

    dictionary.add_token(clskey.fruit, gen.DictItem('사과', unit='개'))
    dictionary.add_token(clskey.fruit, gen.DictItem('배', unit='개'))
    dictionary.add_token(clskey.fruit, gen.DictItem('감', unit='개'))
    dictionary.add_token(clskey.fruit, gen.DictItem('귤', unit='개'))
    dictionary.add_token(clskey.fruit, gen.DictItem('포도', unit='송이'))
    dictionary.add_token(clskey.fruit, gen.DictItem('수박', unit='통'))
    dictionary.add_token(clskey.fruit, gen.DictItem('키위', unit='개'))
    dictionary.add_token(clskey.fruit, gen.DictItem('바나나', unit='개'))
    # dictionary.add_token(clskey.fruit, gen.DictItem('마카다미아', unit='개'))

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
    # dictionary.add_token(clskey.animal, gen.DictItem('강아지', unit='마리'))
    # dictionary.add_token(clskey.animal, gen.DictItem('송아지', unit='마리'))
    # dictionary.add_token(clskey.animal, gen.DictItem('망아지', unit='마리'))
    # dictionary.add_token(clskey.animal, gen.DictItem('병아리', unit='마리'))
    # ...

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

    dictionary.add_token(clskey.place, gen.DictItem('서점'))
    dictionary.add_token(clskey.place, gen.DictItem('마트'))
    dictionary.add_token(clskey.place, gen.DictItem('문구점'))
    dictionary.add_token(clskey.place, gen.DictItem('집'))
    dictionary.add_token(clskey.place, gen.DictItem('학교'))
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

    dictionary.add_token(clskey.name, gen.DictItem('남준'))
    dictionary.add_token(clskey.name, gen.DictItem('석진'))
    dictionary.add_token(clskey.name, gen.DictItem('윤기'))
    dictionary.add_token(clskey.name, gen.DictItem('호석'))
    dictionary.add_token(clskey.name, gen.DictItem('지민'))
    dictionary.add_token(clskey.name, gen.DictItem('태형'))
    dictionary.add_token(clskey.name, gen.DictItem('정국'))
    dictionary.add_token(clskey.name, gen.DictItem('민영'))
    dictionary.add_token(clskey.name, gen.DictItem('유정'))
    dictionary.add_token(clskey.name, gen.DictItem('은지'))
    dictionary.add_token(clskey.name, gen.DictItem('유나'))
    dictionary.add_token(clskey.name, gen.DictItem('철수'))
    dictionary.add_token(clskey.name, gen.DictItem('영희'))
    dictionary.add_token(clskey.name, gen.DictItem('영수'))

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
