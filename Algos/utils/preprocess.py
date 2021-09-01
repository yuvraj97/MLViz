import traceback

import numpy as np


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
