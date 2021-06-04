def process_function(s: str):

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
    return eval(f"lambda {', '.join(args)}: {s}")
