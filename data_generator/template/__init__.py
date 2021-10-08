from .namespace import Namespace
from .tokenregistry import problem_templates, register, TokenRegistry, TokenSelector

clskey = Namespace()


def format(body, question, equation, variable=None, answer=None):
    if variable is None:
        variable = 'ans'
        equation = 'ans = ' + equation

    if answer is None:
        vardict = dict()
        exec('import math, itertools;' + equation, vardict)

        answer = vardict[variable]

    return {
        'body': body,
        'question': question,
        'equation': equation,
        'variable': variable,
        'answer': answer
    }
