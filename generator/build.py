from . import korutil
from . import token
from .equation import equations
from .dictionary import DictItem

import math as _math
import itertools as _itertools

import numbers


def _read_brace(s, i):
    assert s[i] == '{'
    j = i
    while j < len(s) and s[j] != '}':
        j += 1
    return s[i+1:j], j+1


def fnmap(**kwargs):
    return kwargs

def _value(tkn):
    if isinstance(tkn, token.Token):
        return repr(tkn.value)
    if isinstance(tkn, DictItem):
        return repr(tkn.text)
    else:
        return repr(tkn)

def _text(tkn):
    if isinstance(tkn, token.Token):
        return str(tkn.text)
    if isinstance(tkn, DictItem):
        return tkn.text
    else:
        return str(tkn)


def _tokenize(tkn):
    if isinstance(tkn, token.Token):
        return tkn.token
    elif isinstance(tkn, str):
        return tkn
    elif isinstance(tkn, numbers.Number):
        return str(tkn)
    elif isinstance(tkn, list):
        return list(map(_tokenize, tkn))
    elif isinstance(tkn, tuple):
        return tuple(map(_tokenize, tkn))
    elif isinstance(tkn, set):
        return set(map(_tokenize, tkn))
    elif isinstance(tkn, dict):
        return dict(map(
            lambda x: (_tokenize(x[0]), _tokenize(x[1])),
            tkn.items()))


def _format(fstr, env=None, repr=_text):
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
                local = env.copy()
                local['repr'] = repr
                s = str(eval('repr(' + var + ')', local))
                nl_string += s

                item = var.partition('.')[0]
                tk_string += _tokenize(env[item])
                i = ni
        elif fstr[i] == '#':
            assert fstr[i+1] == '{'
            text, i = _read_brace(fstr, i+1)
            c = korutil.append_connection(nl_string[-1], text)
            nl_string += c
            tk_string += c
        else:
            nl_string += fstr[i]
            tk_string += fstr[i]
            i += 1

    return nl_string, tk_string


def build(body, question, equation, variable=None, answer=None, env=None):
    if env is None:
        env = dict()
    body_p, body_k = _format(body, env, repr=_text)
    ques_p, ques_k = _format(question, env, repr=_text)

    eqid = equation.eqnid
    args = equation.args

    equn_p = '{}({})'.format(eqid, ', '.join(map(str, map(_value, args))))
    equn_k = '{} {}'.format(eqid, ' '.join(map(_tokenize, args)))

    if answer is None:
        local = dict()
        local[equation.eqnid] = equations.get(equation.eqnid)
        eqn = eval(equn_p, local)

        local = dict()
        answer = exec('import math, itertools\n' + eqn, local)
        answer = local['ans']

    return fnmap(
            parsed=fnmap(
                body=body_p,
                question=ques_p,
                equation=equn_p,
                answer=answer
                ),
            token=fnmap(
                body=body_k,
                question=ques_k,
                equation=equn_k
                ),
            )
