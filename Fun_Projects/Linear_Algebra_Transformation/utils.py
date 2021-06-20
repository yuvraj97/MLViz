from typing import Tuple

import numpy as np


def str2matrix(s: str) -> Tuple[np.ndarray, bool]:
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
