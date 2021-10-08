equations = []


def equation(func):
    def _inject_params():
        pass
    equations.append(func)
    return func


@equation
def equation_aaa():
    return ""
