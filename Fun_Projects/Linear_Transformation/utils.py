from typing import Tuple, Union
import numpy as np
import streamlit as st


def str2matrix(s: str) -> Tuple[Union[np.ndarray, None], bool]:
    remove = {' ', '[', ']', '(', ')', '\n'}
    allowed = {',', '-', ';', '.'}
    new_s = "".join([c for c in s if c not in remove])
    for c in new_s:
        if not c.isnumeric() and c not in allowed:
            return None, False

    matrix = np.array([
        [float(element) for element in row.split(",")]
        for row in new_s.split(";")
    ])

    return matrix, True

def validate_equation(equation: str):
    equation = equation.lower().replace("[", "(").replace("]", ")")
    allowed_symbols = {'x', 'y', '+', '-', '*', '/', '^', '(', ')', ' '}
    result, idx = [], 0
    while idx < len(equation):
        char = equation[idx]
        if char in allowed_symbols or char.isnumeric(): result.append(char)
        elif char.isalpha():
            result.append("np.")
            cmd = []
            while idx < len(equation) and equation[idx].isalpha():
                cmd.append(equation[idx])
                idx += 1
            idx -= 1
            cmd = "".join(cmd)
            if cmd in dir(np.math):
                result.append(cmd)
            else:
                st.write(f"CMD: {cmd}")
                return None, False
        else:
            st.write(f"Char: {char}")
            return None, False
        idx += 1

    try:
        args = "x, y" if "y" in result else "x"
        print(f'lambda {args}: {"".join(result).replace("^", "**")}')
        return eval(f'lambda {args}: {"".join(result).replace("^", "**")}'), True
    except:
        return None, False

def str2vec(s: str):
    allowed = {',', '-', '.'}
    new_s = "".join([c for c in s if c.isnumeric() or c in allowed])
    vector = [float(element) for element in new_s.split(",")]
    return vector, True

def vec2str(vector: np.ndarray):
    return "[" + ", ".join([str(e) for e in vector]) + "]"
