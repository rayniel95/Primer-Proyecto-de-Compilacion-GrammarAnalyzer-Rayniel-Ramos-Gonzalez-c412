import math


def tokenize(expr):
    """
    Returns the set of tokens. At this point, simply splits by
    spaces and converts numbers to `float` instances.
    Replaces constants.
    """
    tokens = []

    for token in expr.split():
        try:
            tokens.append(float(token))
        except:
            if (token in constants):
                tokens.append(constants[token])
            else:
                tokens.append(token)

    return tokens

operations = {
    '+': lambda x,y: x + y,
    '-': lambda x,y: x - y,
    '*': lambda x,y: x * y,
    '/': lambda x,y: x / y,
}

constants = {
    'pi': 3.14159265359,
    'e': 2.71828182846,
    'phi': 1.61803398875,
}

functions = {
    'sin': lambda x: math.sin(x),
    'cos': lambda x: math.cos(x),
    'tan': lambda x: math.tan(x),
    'log': lambda x,y: math.log(x, y),
    'sqrt': lambda x: math.sqrt(x),
}

# region Errors Class

class ParsingError(Exception):
    """
    Base class for all parsing exceptions.
    """
    pass


class BadEOFError(ParsingError):
    """
    Unexpected EOF error.
    """

    def __init__(self):
        ParsingError.__init__(self, "Unexpected EOF")


class UnexpectedToken(ParsingError):
    """
    Unexpected token error.
    """

    def __init__(self, token, i):
        ParsingError.__init__(f'Unexpected token: {token} at {i}')


class MissingCloseParenthesisError(ParsingError):
    """
    Missing ')' token error.
    """

    def __init__(self, token, i):
        ParsingError.__init__(
            f'Expected ")" token at {i}. Got "{token}" instead')


class MissingOpenParenthesisError(ParsingError):
    """
    Missing '(' token error.
    """

    def __init__(self, token, i):
        ParsingError.__init__(
            f'Expected "(" token at {i}. Got "{token}" instead')

# endregion

def get_token(tokens, i, error_type=BadEOFError):
    """
    Returns tokens[i] if 'i' is in range. Otherwise, raises ParsingError
    exception.
    """
    try:
        return tokens[i]
    except IndexError:
        raise error_type()

def evaluate(tokens):
    """
    Evaluates an expression recursively.
    """
    try:
        i, value = parse_expression(tokens, 0)
        assert i == len(tokens)
        return value
    except ParsingError as error:
        print(error)
        return None


def parse_expression(tokens, i):
    i, term = parse_term(tokens, i)

    if i < len(tokens):
        if tokens[i] == '+' or tokens[i] == '-':
            # Insert your code here ...
            ##################

            other_index, other_term = parse_term(tokens, i + 1)
            term = operations[tokens[i]] (term, other_term)

            return other_index, term

            ##################
    return i, term


def parse_term(tokens, i) -> tuple:
    # Insert your code here ...
    ##################
    i, term = parse_factor(tokens, i)

    if i < len(tokens):
        if tokens[i] == '*' or tokens[i] == '/':
            other_index, other_term = parse_factor(tokens, i + 1)
            term = operations[tokens[i]](term, other_term)

            return other_index, term

        if tokens[i] == '+' or tokens[i] == '-':
            other_index, other_term = parse_term(tokens, i + 1)
            term = operations[tokens[i]](term, other_term)

            return other_index, term

    return i, term
    ##################


def parse_factor(tokens, i) -> tuple:
    # Insert your code here ...
    ##################
    if isinstance(tokens[i], float):
        return i + 1, tokens[i]

    if tokens[i] == '(':
        other_index, other_term = parse_expression(tokens, i + 1)

        return other_index + 1, other_term

    if tokens[i] in functions:

        first_argument_index, first_argument = parse_expression(tokens, i + 2)

        second_argument_index, second_argument = \
            parse_expression(tokens, first_argument_index + 1)

        return second_argument_index + 1, functions[tokens[i]] (first_argument,
                                                                second_argument)
    ##################

assert evaluate(tokenize('5 + 6 * 9')) == 59.
assert evaluate(tokenize('( 5 + 6 ) * 9')) == 99.
assert evaluate(tokenize('( 5 + 6 ) + 1 * 9 + 2')) == 22.
assert tokenize('5 + 6 * 9') == [5, '+', 6, '*', 9]
assert tokenize('2 * pi') == [2.0, '*', 3.14159265359]
assert evaluate(tokenize('2 * pi')) == 6.28318530718
assert tokenize('log ( 64 , 1 + 3 )') == ['log', '(', 64.0, ',', 1.0, '+', 3.0,
                                          ')']
assert evaluate(tokenize('log ( 64 , 1 + 3 )')) == 3.0