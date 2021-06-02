import enum
from enum import Enum

# todo falta hacer el arbol de derivacion, para ello modificar los metodos para
# pasar los nodos del arbol y este se vaya creando de manera recursiva junto al
# llamado de los metodos

TokenType = Enum('TokenType', 'eof num plus minus star div opar cpar id')


class Token:
    """
    Basic token class.

    Parameters
    ----------
    lex : str
        Token's lexeme.
    token_type : Enum
        Token's type.
    """

    def __init__(self, lex: str, token_type: enum.Enum):
        self.lex = lex
        self.token_type = token_type


EOF_TOKEN = Token('$', TokenType.eof)

fixed_tokens = {
    '+'  :   Token( '+'           , TokenType.plus  ),
    '-'  :   Token( '-'           , TokenType.minus ),
    '*'  :   Token( '*'           , TokenType.star  ),
    '/'  :   Token( '/'           , TokenType.div   ),
    '('  :   Token( '('           , TokenType.opar  ),
    ')'  :   Token( ')'           , TokenType.cpar  ),
    'pi' :   Token( 3.14159265359 , TokenType.num   ),
    'e'  :   Token( 2.71828182846 , TokenType.num   ),
    'phi':   Token( 1.61803398875 , TokenType.num   ),
}


class ParsingError(Exception):
    """
    Base class for all parsing exceptions.
    """
    pass


class Lexer:
    """
    Base lexer class.

    Parameters
    ----------
    text : str
        String to tokenize.
    """

    def __init__(self, text: str):
        self.index = 0
        self.text = text
        self.tokens = self.tokenize_text()

    def tokenize_text(self):
        """
        Tokenize `self.text` and set it to `self.tokens`.
        """
        raise NotImplementedError()

    def next_token(self):
        """
        Returns the next tokens readed by the lexer. `None` if `self.tokens` is
        exhausted.
        """
        try:
            token = self.tokens[self.index]
            self.index += 1
            return token
        except IndexError:
            return None

    def is_done(self):
        """
        Returns whether or not `self.tokens` is exhausted.
        """
        try:
            self.tokens[self.index]
            return False
        except IndexError:
            return True


class Parser:
    """
    Base parser class.
    """

    def __init__(self):
        self.lexer = None
        self.left_parse: list = None
        self.lookahead: enum.Enum = None

    def parse(self, lexer):
        """
        Returns a left parse given the tokens from the lexer.
        """
        try:
            self.lexer = lexer
            self.left_parse = []
            self.lookahead = lexer.next_token().token_type
            self.begin()
            return self.left_parse

        except ParsingError as error:
            print('Parsing error: {error}!!!'.format(**locals()))
            print('Lookahead: {self.lookahead}'.format(**locals()))
            print('Unfinished parse: {self.left_parse}'.format(**locals()))

        finally:
            self.lex = None
            self.left_parse = None
            self.lookahead = None

    def begin(self):
        """
        Begin parsing from starting symbol and match EOF.
        """
        raise NotImplementedError()

    def report(self, production):
        """
        Adds production to the left parse that is being build.
        """
        self.left_parse.append(production)

    def error(self, msg=None):
        """
        Raises a parsing error.
        """
        raise ParsingError(msg)

    def match(self, token_type):
        """
        Consumes one token from the lexer if lookahead matches the given token type.
        Raises parsing error otherwise.
        """
        if token_type == self.lookahead:
            try:
                self.lookahead = self.lexer.next_token().token_type
            except AttributeError:
                self.lookahead = None
        else:
            self.error('Unexpected token')



class XCoolLexer(Lexer):

    def tokenize_text(self):
        tokens = []
        text = self.text

        for item in text.split():
            try:
                tokens.append(Token(float(item), TokenType.num))
            except ValueError:
                if item in fixed_tokens:
                    tokens.append(Token(item, fixed_tokens[item]).token_type)
                else:
                    raise ParsingError('token not exist')

        # Is something missing?
        tokens.append(EOF_TOKEN)

        # que cosa es el token de tipo id?

        return tokens


class XCoolParser(Parser):

    class Node:

        def __init__(self, value, parent=None):
            self._value = value
            self._parent = parent
            self.sons: list = []

        @property
        def parent(self):
            return self._parent

    def begin(self):
        self.E()
        self.match(TokenType.eof)

    def E(self):
        """
        E --> TX
        """
        if self.lookahead in (TokenType.num, TokenType.opar):
            self.report('E --> TX')
            self.T()
            self.X()

        else:
            self.error('Malformed expression')

    def X(self):
        """
        X --> +TX | -TX | epsilon
        """
        # Insert your code here ...
        if self.lookahead == TokenType.plus:
            self.report('X --> +TX')
            self.match(TokenType.plus)
            self.T()
            self.X()
        elif self.lookahead == TokenType.minus:
            self.report('X --> -TX')
            self.match(TokenType.minus)
            self.T()
            self.X()
        elif self.lookahead in (TokenType.eof, TokenType.div, TokenType.star):
            self.report('X --> epsilon')

        else:
            self.error('Malformed expression')

    def T(self):
        """
        T --> FY
        """
        # Insert your code here ...
        self.report('T --> FY')
        self.F()
        self.Y()

    def Y(self):
        """
        Y --> *FY | /FY | epsilon
        """
        # Insert your code here ...
        if self.lookahead == TokenType.star:
            self.report('Y --> *FY')
            self.match(TokenType.star)
            self.F()
            self.Y()
        elif self.lookahead == TokenType.div:
            self.report('Y --> /FY')
            self.match(TokenType.div)
            self.F()
            self.Y()
        elif self.lookahead in (TokenType.eof, TokenType.plus, TokenType.minus):
            self.report('Y --> epsilon')
        else:
            self.error('Malformed expression')

    def F(self):
        """
        F --> n | (E)
        """
        # Insert your code here ...
        if self.lookahead == TokenType.num:
            self.report('F --> n')
            self.match(TokenType.num)
        elif self.lookahead == TokenType.opar:
            self.report('F --> (E)')
            self.match(TokenType.opar)
            self.E()
            if self.lookahead == TokenType.cpar:
                self.match(TokenType.cpar)
            else:
                self.error('Malformed expression')
        else:
            self.error('Malformed expression')


def get_left_parse(text):
    lexer = XCoolLexer(text)
    parser = XCoolParser()
    return parser.parse(lexer)

assert get_left_parse('5 + 8 * 9') == [  'E --> TX',
                                         'T --> FY',
                                         'F --> n',
                                         'Y --> epsilon',
                                         'X --> +TX',
                                         'T --> FY',
                                         'F --> n',
                                         'Y --> *FY',
                                         'F --> n',
                                         'Y --> epsilon',
                                         'X --> epsilon'  ]
print(get_left_parse('1 - 1 + 1'))
assert get_left_parse('1 - 1 + 1') == [  'E --> TX',
                                         'T --> FY',
                                         'F --> n',
                                         'Y --> epsilon',
                                         'X --> -TX',
                                         'T --> FY',
                                         'F --> n',
                                         'Y --> epsilon',
                                         'X --> +TX',
                                         'T --> FY',
                                         'F --> n',
                                         'Y --> epsilon',
                                         'X --> epsilon'  ]