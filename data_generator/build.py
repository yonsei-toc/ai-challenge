def build_problems():
    import data_generator.all_problems as all_problems
    import data_generator.template as template
    from data_generator.template import clskey
    problem_fns = template.problem_templates.problem_fns
    dictionary = template.TokenRegistry()
    all_problems.setup_dictionary(clskey, dictionary)

    def _generate_problem(problem_id):
        problem_fn = problem_fns[problem_id]
        tokens = template.TokenSelector(dictionary)
        return problem_fn(tokens)
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
