from generator import korutil
from generator import token
from data.equations import equations
from generator.dictionary import DictItem

import math as _math
import itertools as _itertools

import numbers


def _read_brace(s, i):
    assert s[i] == '{'
    j = s.find('}', i)
    assert j != -1
    return s[i + 1:j], j + 1


def fnmap(**kwargs):
    return kwargs


# expr context
def _value(tkn):
    if isinstance(tkn, token.Token):
        return repr(tkn.value)
    elif isinstance(tkn, DictItem):
        return repr(tkn.text)
    else:
        return repr(tkn)


# text context
def _text(tkn):
    if isinstance(tkn, token.Token):
        return str(tkn.text)
    elif isinstance(tkn, DictItem):
        return tkn.text
    else:
        return str(tkn)


# token context
def _token(tkn):
    if isinstance(tkn, token.Token):
        return tkn.token
    elif isinstance(tkn, DictItem):
        return tkn.text
    elif isinstance(tkn, str):
        return tkn
    elif isinstance(tkn, numbers.Number):
        return str(tkn)
    elif isinstance(tkn, list):
        return '[{}]'.format(' '.join(map(_token, tkn)))
    elif isinstance(tkn, tuple):
        return '({})'.format(' '.join(map(_token, tkn)))
    elif isinstance(tkn, set):
        return '{' + ' '.join(map(_token, tkn)) + '}'
    elif isinstance(tkn, dict):
        return '{' + ' '.join(map(lambda x: '{}:{}'.format(_token(x[0]), _token(x[1])), tkn.items())) + '}'


def _format(fstr, env=None):
    # {item} = repr(item)
    # {item.sdfs} = eval(...)
    # #{가} = (on nl_string) append_connection(s[-1], "가")
    # repr = repr for equation
    nl_string = ''
    tk_string = ''
    i = 0

    while i < len(fstr):
        if fstr[i] == '{':
            if fstr[i + 1] == '{':
                nl_string += '{'
                tk_string += '{'
                i += 2
            else:
                # TODO: handle aggregations
                var, ni = _read_brace(fstr, i)

                if var[0] == '#':
                    c = korutil.append_connection(nl_string[-1], var[1:])
                    if c is not None:
                        nl_string += c
                        tk_string += c
                    else:
                        nl_string += var[1:]
                        tk_string += var[1:]
                else:
                    local = env.copy()
                    local['repr'] = _text
                    s = str(eval('repr(' + var + ')', local))
                    nl_string += s

                    item = var.partition('.')[0]
                    if isinstance(env[item], token.Token):
                        tk_string += _token(env[item])
                    else:
                        tk_string += s
                i = ni
        else:
            nl_string += fstr[i]
            tk_string += fstr[i]
            i += 1

    return nl_string, tk_string


def build(body, question, equation, variable=None, answer=None, env=None):
    if env is None:
        env = dict()
    body_p, body_k = _format(body, env)
    ques_p, ques_k = _format(question, env)

    eqid = equation.eqnid
    args = equation.args

    equn_p = '{}({})'.format(eqid, ', '.join(map(str, map(_value, args))))
    equn_k = '{} {}'.format(eqid, ' '.join(map(_token, args)))

    local = {
        equation.eqnid: equations.get(equation.eqnid)
    }
    eqn = eval(equn_p, local)

    # if answer is None:
    _l = {}
    exec(eqn, _l)
    answer = _l['ans']

    return fnmap(
        origin_body=body_p,
        origin_question=ques_p,
        token_body=body_k,
        token_question=ques_k,
        tokens={arg.token: arg for arg in args if isinstance(arg, token.Token)},
        origin_equation=equn_p,
        token_equation=equn_k,
        equation_type=equations.get_id(eqid),
        equation_tokens=[arg.token if isinstance(arg, token.Token) else arg for arg in args],
        equation_args=args,
        answer=answer,
        code=eqn
    )


def build_problems():
    import generator.dictionary as gd
    import generator.problem as gp
    import generator.token as gt
    import generator.dictselector as gds
    import script

    with open('script/dict.json', 'rt', encoding='utf8') as f:
        dictionary, clskey = gd.Dictionary.load(f.read())

    problem_fns = gp.problems

    def _generate_problem(problem_id):
        selector = gds.DictionarySelector(dictionary)
        problem_fn = problem_fns[problem_id]
        token_pool = gt.TokenPool()
        return problem_fn(selector, token_pool, clskey)

    return _generate_problem, list(range(len(problem_fns)))


if __name__ == '__main__':
    prob_fn, prob_ids = build_problems()
    for prob_id in prob_ids:
        i = 0
        while i < 3:
            ret = prob_fn(prob_id)
            if ret is None:
                continue
            else:
                i += 1
                print(ret)
