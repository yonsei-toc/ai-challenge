from . import korutil
from . import token
from .equation import equations
from .dictionary import DictItem

import math as _math
import itertools as _itertools

import numbers


def _read_brace(s, i):
    assert s[i] == '{'
    j = s.find('}', i)
    assert j != -1
    return s[i+1:j], j+1


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
            if fstr[i+1] == '{':
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
                    tk_string += _token(env[item])
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

    if answer is None:
        local = dict()
        answer = exec('import math, itertools\n' + eqn, local)
        answer = local['ans']

    return fnmap(
            parsed=fnmap(
                body=body_p,
                question=ques_p,
                equation=equn_p,
                answer=answer,
                code=eqn
                ),
            token=fnmap(
                body=body_k,
                question=ques_k,
                equation=equn_k
                ),
            )
