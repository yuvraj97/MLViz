import traceback

import numpy as np


def process_function_0(s: str):
    """
    Here we take a function expression as input then first we validate it
    then we return a lambda function corresponding to that expression
    :param s: str
    :return:
    """

    if s == "": return False
    allowed = ["+", "-", "*", "/", "**", " "]
    s = s.replace("^", "**")

    for c in s:
        if not c.isnumeric() and c not in allowed and c != 'x':
            return False

    _s = s
    for c in allowed:
        _s = _s.replace(c, " ")
    s_list = _s.split(" ")

    args = set()
    for e in s_list:
        if e == "": continue
        if 'x' == e[0]:
            args.add(e)
    args = sorted(args)
    try:
        return eval(f"lambda {', '.join(args)}: {s}")
    except Exception as e:
        print(e)
        return False


allowed_functions = set(dir(np.math))


def process_function(equation: str):
    equation = equation.lower().replace("[", "(").replace("]", ")")
    allowed_symbols = {'x', '+', '-', '*', '/', '^', '(', ')', ' '}
    result, idx = [], 0
    while idx < len(equation):
        char = equation[idx]
        if char in allowed_symbols or char.isnumeric():
            result.append(char)
        elif char.isalpha():
            result.append("np.")
            cmd = []
            while idx < len(equation) and equation[idx].isalpha():
                cmd.append(equation[idx])
                idx += 1
            idx -= 1
            cmd = "".join(cmd)
            if cmd in allowed_functions:
                result.append(cmd)
            else:
                print(f"CMD: {cmd}")
                return None
        else:
            print(f"Char: {char}")
            return None
        idx += 1

    try:
        args = set()
        for i, c in enumerate(equation):
            c_next = equation[i + 1] if i + 1 < len(equation) else None
            if c.lower() == "x" and (c_next and c_next.isnumeric()):
                args.add(c + c_next)
        args = sorted(args)
        return eval(f'lambda {", ".join(args)}: {"".join(result).replace("^", "**")}')
    except Exception as e:
        traceback.print_exc()
        return None
