import json
from itertools import islice


class Production(object):

    def __init__(self, nonTerminal, sentence):
        self.Left = nonTerminal
        self.Right = sentence

    def __str__(self):
        return '%s := %s' % (self.Left, self.Right)

    def __repr__(self):
        return '%s -> %s' % (self.Left, self.Right)

    def __iter__(self):
        yield self.Left
        yield self.Right

    def __eq__(self, other):
        return isinstance(other, Production) and self.Left == other.Left and \
               self.Right == other.Right

    @property
    def IsEpsilon(self):
        return self.Right.IsEpsilon


class AttributeProduction(Production):

    def __init__(self, nonTerminal, sentence, attributes):
        if not isinstance(sentence, Sentence) and isinstance(sentence, Symbol):
            sentence = Sentence(sentence)
        super(AttributeProduction, self).__init__(nonTerminal, sentence)

        self.attributes = attributes

    def __str__(self):
        return '%s := %s' % (self.Left, self.Right)

    def __repr__(self):
        return '%s -> %s' % (self.Left, self.Right)

    def __iter__(self):
        yield self.Left
        yield self.Right

    @property
    def IsEpsilon(self):
        return self.Right.IsEpsilon


class Symbol(object):
    '''
    no se debe instanciar a partir del constructor, o sea, no se debe instanciar
    '''
    def __init__(self, name, grammar):
        self.Name = name
        self.Grammar = grammar

    def __str__(self):
        return self.Name

    def __repr__(self):
        return repr(self.Name)

    def __add__(self, other):
        if isinstance(other, Symbol):
            return Sentence(self, other)

        raise TypeError(other)

    def __or__(self, other):

        if isinstance(other, (Sentence)):
            return SentenceList(Sentence(self), other)

        raise TypeError(other)

    @property
    def IsEpsilon(self):
        return False

    def __len__(self):
        return 1

class NonTerminal(Symbol):
    '''
    no se debe instanciar
    '''
    def __init__(self, name, grammar):
        super().__init__(name, grammar)
        self.productions = []

    def __imod__(self, other):

        if isinstance(other, (Sentence)):
            p = Production(self, other)
            self.Grammar.Add_Production(p)
            return self

        if isinstance(other, tuple):
            assert len(other) == 2, "Tiene que ser una Tupla de 2 elementos " \
                                    "(sentence, attribute)"

            if isinstance(other[0], Symbol):
                p = AttributeProduction(self, Sentence(other[0]), other[1])
            elif isinstance(other[0], Sentence):
                p = AttributeProduction(self, other[0], other[1])
            else:
                raise Exception("")

            self.Grammar.Add_Production(p)
            return self

        if isinstance(other, Symbol):
            p = Production(self, Sentence(other))
            self.Grammar.Add_Production(p)
            return self

        if isinstance(other, SentenceList):

            for s in other:
                p = Production(self, s)
                self.Grammar.Add_Production(p)

            return self

        raise TypeError(other)

    @property
    def IsTerminal(self):
        return False

    @property
    def IsNonTerminal(self):
        return True

    @property
    def IsEpsilon(self):
        return False


class Terminal(Symbol):

    def __init__(self, name, grammar):
        super().__init__(name, grammar)

    @property
    def IsTerminal(self):
        return True

    @property
    def IsNonTerminal(self):
        return False

    @property
    def IsEpsilon(self):
        return False


class EOF(Terminal):

    def __init__(self, Grammar):
        super().__init__('$', Grammar)


class Sentence(object):

    def __init__(self, *args):
        self._symbols = tuple(x for x in args if not x.IsEpsilon)
        self.hash = hash(self._symbols)

    def __len__(self):
        return len(self._symbols)

    def __add__(self, other):
        if isinstance(other, Symbol):
            return Sentence(*(self._symbols + (other,)))

        if isinstance(other, Sentence):
            return Sentence(*(self._symbols + other._symbols))

        raise TypeError(other)

    def __or__(self, other):
        if isinstance(other, Sentence):
            return SentenceList(self, other)

        if isinstance(other, Symbol):
            return SentenceList(self, Sentence(other))

        raise TypeError(other)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return ("%s " * len(self._symbols) % tuple(self._symbols)).strip()

    def __iter__(self):
        return iter(self._symbols)

    def __getitem__(self, index):
        return self._symbols[index]

    def __eq__(self, other):
        return self._symbols == other._symbols

    def __hash__(self):
        return self.hash

    @property
    def IsEpsilon(self):
        return False


class SentenceList(object):

    def __init__(self, *args):
        self._sentences = list(args)

    def Add(self, symbol):
        if not symbol and (symbol is None or not symbol.IsEpsilon):
            raise ValueError(symbol)

        self._sentences.append(symbol)

    def __iter__(self):
        return iter(self._sentences)

    def __or__(self, other):
        if isinstance(other, Sentence):
            self.Add(other)
            return self

        if isinstance(other, Symbol):
            return self | Sentence(other)


class Epsilon(Terminal, Sentence):

    def __init__(self, grammar):
        super().__init__('epsilon', grammar)

    def __str__(self):
        return "e"

    def __repr__(self):
        return 'epsilon'

    def __iter__(self):
        yield self

    def __len__(self):
        return 0

    def __add__(self, other):
        return other

    def __eq__(self, other):
        return isinstance(other, (Epsilon,))

    def __hash__(self):
        return hash("")

    @property
    def IsEpsilon(self):
        return True


class Grammar():

    def __init__(self):

        self.Productions = []
        self.nonTerminals = []
        self.terminals = []
        self.startSymbol = None
        # production type
        self.pType = None
        self.Epsilon = Epsilon(self)
        self.EOF = EOF(self)

        self.symbDict = {}

    def NonTerminal(self, name, startSymbol = False):

        name = name.strip()
        if not name:
            raise Exception("Empty name")

        term = NonTerminal(name,self)

        if startSymbol:

            if self.startSymbol is None:
                self.startSymbol = term
            else:
                raise Exception("Cannot define more than one start symbol.")

        self.nonTerminals.append(term)
        self.symbDict[name] = term
        return term

    def NonTerminals(self, names):

        ans = tuple((self.NonTerminal(x) for x in names.strip().split()))

        return ans


    def Add_Production(self, production):

        if len(self.Productions) == 0:
            self.pType = type(production)

        assert type(production) == self.pType, "The Productions most be of only 1 type."

        production.Left.productions.append(production)
        self.Productions.append(production)


    def Terminal(self, name):

        name = name.strip()
        if not name:
            raise Exception("Empty name")

        term = Terminal(name, self)
        self.terminals.append(term)
        self.symbDict[name] = term
        return term

    def Terminals(self, names):

        ans = tuple((self.Terminal(x) for x in names.strip().split()))

        return ans


    def __str__(self):

        mul = '%s, '

        ans = 'Non-Terminals:\n\t'

        nonterminals = mul * (len(self.nonTerminals)-1) + '%s\n'

        ans += nonterminals % tuple(self.nonTerminals)

        ans += 'Terminals:\n\t'

        terminals = mul * (len(self.terminals)-1) + '%s\n'

        ans += terminals % tuple(self.terminals)

        ans += 'Productions:\n\t'

        ans += str(self.Productions)

        return ans

    @property
    def to_json(self):

        productions = []

        for p in self.Productions:
            head = p.Left.Name

            body = []

            for s in p.Right:
                body.append(s.Name)

            productions.append({'Head':head, 'Body':body})

        d={'NonTerminals':[symb.Name for symb in self.nonTerminals], 'Terminals': [symb.Name for symb in self.terminals],\
         'Productions':productions}

         # [{'Head':p.Left.Name, "Body": [s.Name for s in p.Right]} for p in self.Productions]
        return json.dumps(d)

    @staticmethod
    def from_json(data):
        data = json.loads(data)

        G = Grammar()
        dic = {'epsilon':G.Epsilon}

        for term in data['Terminals']:
            dic[term] = G.Terminal(term)

        for noTerm in data['NonTerminals']:
            dic[noTerm] = G.NonTerminal(noTerm)

        for p in data['Productions']:
            head = p['Head']
            dic[head] %= Sentence(*[dic[term] for term in p['Body']])

        return G

    def copy(self):
        G = Grammar()
        G.Productions = self.Productions.copy()
        G.nonTerminals = self.nonTerminals.copy()
        G.terminals = self.terminals.copy()
        G.pType = self.pType
        G.startSymbol = self.startSymbol
        G.Epsilon = self.Epsilon
        G.EOF = self.EOF
        G.symbDict = self.symbDict.copy()

        return G

    @property
    def IsAugmentedGrammar(self):
        augmented = 0
        for left, right in self.Productions:
            if self.startSymbol == left:
                augmented += 1
        if augmented <= 1:
            return True
        else:
            return False

    def AugmentedGrammar(self):
        if not self.IsAugmentedGrammar:

            G = self.copy()
            # S, self.startSymbol, SS = self.startSymbol, None, self.NonTerminal
            # ('S\'', True)
            S = G.startSymbol
            G.startSymbol = None
            SS = G.NonTerminal('S\'', True)
            if G.pType is AttributeProduction:
                SS %= S + G.Epsilon, lambda x : x
            else:
                SS %= S + G.Epsilon

            return G
        else:
            return self.copy()
    #endchange


class ContainerSet:
    def __init__(self, *values, contains_epsilon=False):
        self.set = set(values)
        self.contains_epsilon = contains_epsilon

    def add(self, value):
        n = len(self.set)
        self.set.add(value)
        return n != len(self.set)

    def set_epsilon(self, value=True):
        last = self.contains_epsilon
        self.contains_epsilon = value
        return last != self.contains_epsilon

    def update(self, other):
        n = len(self.set)
        self.set.update(other.set)
        return n != len(self.set)

    def epsilon_update(self, other):
        return self.set_epsilon(self.contains_epsilon | other.contains_epsilon)

    def hard_update(self, other):
        return self.update(other) | self.epsilon_update(other)

    def __len__(self):
        return len(self.set) + int(self.contains_epsilon)

    def __str__(self):
        return '%s-%s' % (str(self.set), self.contains_epsilon)

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self.set)

    def __eq__(self, other):
        return isinstance(other, ContainerSet) and self.set == other.set and \
               self.contains_epsilon == other.contains_epsilon


# Computes First(alpha), given First(Vt) and First(Vn)
# alpha in (Vt U Vn)*
def compute_local_first(firsts, alpha):
    first_alpha = ContainerSet()

    try:
        alpha_is_epsilon = alpha.IsEpsilon
    except:
        alpha_is_epsilon = False

    ###################################################
    # alpha == epsilon ? First(alpha) = { epsilon }
    ###################################################
    #                   <CODE_HERE>
    if alpha_is_epsilon:
        first_alpha.set_epsilon()
        return first_alpha
    ###################################################

    ###################################################
    # alpha = X1 ... XN
    # First(Xi) subconjunto First(alpha)
    # epsilon pertenece a First(X1)...First(Xi) ? First(Xi+1) subconjunto de First(X) y First(alpha)
    # epsilon pertenece a First(X1)...First(XN) ? epsilon pertence a First(X) y al First(alpha)
    ###################################################
    #                   <CODE_HERE>                   #
    for symbol in alpha:
        if isinstance(symbol, Terminal):
            first_alpha.add(symbol)
            return first_alpha
            # estoy asumiendo q no apareceran terminales en la parte central o
            # derecha de la forma oracional, es esto correcto para gramaticas
            # LL(1)????????????
        first_alpha.update(firsts[symbol])

        if not firsts[symbol].contains_epsilon:
            return first_alpha
    ###################################################
    first_alpha.set_epsilon()
    # First(alpha)
    return first_alpha

# Computes First(Vt) U First(Vn) U First(alpha)
# P: X -> alpha
def compute_firsts(G):
    firsts = {}
    change = True

    # init First(Vt)
    for terminal in G.terminals:
        firsts[terminal] = ContainerSet(terminal)

    # init First(Vn)
    for nonterminal in G.nonTerminals:
        firsts[nonterminal] = ContainerSet()

    while change:
        change = False

        # P: X -> alpha
        for production in G.Productions:
            X = production.Left
            alpha = production.Right

            # get current First(X)
            first_X = firsts[X]

            # init First(alpha)
            try:
                first_alpha = firsts[alpha]
            except:
                first_alpha = firsts[alpha] = ContainerSet()

            # CurrentFirst(alpha)???
            local_first = compute_local_first(firsts, alpha)

            # update First(X) and First(alpha) from CurrentFirst(alpha)
            change |= first_alpha.hard_update(local_first)
            change |= first_X.hard_update(local_first)

    # First(Vt) + First(Vt) + First(RightSides)
    return firsts


def compute_follows(G, firsts: dict):
    follows = {}
    change = True

    local_firsts = {}

    # init Follow(Vn)
    for nonterminal in G.nonTerminals:
        follows[nonterminal] = ContainerSet()
    follows[G.startSymbol] = ContainerSet(G.EOF)

    while change:
        change = False

        # P: X -> alpha
        for production in G.Productions:
            X = production.Left
            alpha = production.Right

            follow_X = follows[X]

            ###################################################
            # X -> zeta Y beta
            # First(beta) - { epsilon } subset of Follow(Y)
            # beta ->* epsilon or X -> zeta Y ? Follow(X) subset of Follow(Y)
            ###################################################
            #                   <CODE_HERE>                   #
            for index in range(len(alpha)):
                no_terminal = alpha[index]

                if isinstance(no_terminal, NonTerminal):
                    if index < len(alpha) - 1:
                        rest = islice(alpha, index + 1, len(alpha))

                        if firsts.get(rest, False):
                            first_rest = firsts[rest]
                        elif local_firsts.get(rest, False):
                            first_rest = local_firsts[rest]

                        else:
                            first_rest = compute_local_first(firsts, rest)
                            local_firsts[rest] = first_rest

                        if follows[no_terminal].update(first_rest):
                            change = True
                        if first_rest.contains_epsilon:
                            if follows[no_terminal].update(follow_X):
                                change = True

                    else:
                        if follows[no_terminal].update(follow_X): change = True
            ###################################################

    # Follow(Vn)
    return follows


def build_parsing_table(G: Grammar, firsts: dict, follows: dict):
    # init parsing table
    M = {}

    # P: X -> alpha
    for production in G.Productions:
        X = production.Left
        alpha = production.Right

        ###################################################
        # working with symbols on First(alpha) ...
        ###################################################
        #                   <CODE_HERE>                   #
        for terminal in G.terminals:
            if terminal in firsts[alpha]:
                if M.get((X, terminal), False):
                    return {}
                # tener presente q aqi se trabaja con listas
                M[X, terminal] = [production]

            elif firsts[alpha].contains_epsilon and terminal in follows[X]:
                if M.get((X, terminal), False):
                    return {}

                M[X, terminal] = [production]

        if G.EOF in firsts[alpha]:
            if M.get((X, G.EOF), False):
                return {}

            M[X, G.EOF] = [production]

        elif firsts[alpha].contains_epsilon and G.EOF in follows[X]:
            if M.get((X, G.EOF), False):
                return {}

            M[X, G.EOF] = [production]
        ###################################################

    # parsing table is ready!!!
    return M


def metodo_predictivo_no_recursivo(G, M=None, firsts=None, follows=None):
    # checking table...
    if M is None:
        if firsts is None:
            firsts = compute_firsts(G)
        if follows is None:
            follows = compute_follows(G, firsts)
        M = build_parsing_table(G, firsts, follows)

    # parser construction...
    def parser(w: list):

        ###################################################
        # w ends with $ (G.EOF)
        ###################################################
        # init:
        stack = [G.EOF, G.startSymbol]
        cursor = 0
        output = []  # las producciones aplicadas
        ###################################################

        # parsing w...
        while True:
            top = stack.pop()
            a = w[cursor]

        #             print((top, a))

        ###################################################
        #                   <CODE_HERE>                   #
        ###################################################
            if isinstance(top, Terminal):
                if a != top: return []
                cursor += 1

                if cursor >= len(w): break

            else:
                if not M.get((top, a), False): return []
                output.append(M[top, a][0])

                sentence: Sentence = M[top, a][0].Right
                for symbol in reversed(sentence):
                    stack.append(symbol)
        # left parse is ready!!!
        return output

    # parser is ready!!!
    return parser

# todo ver lo de las listas a la hora de llenar la tabla si es necesario
#      meterlo en lista


algo = Symbol('name', Grammar())

while algo.IsEpsilon:
    continue


# region test1

# G = Grammar()
# E = G.NonTerminal('E', True)
# T,F,X,Y = G.NonTerminals('T F X Y')
# plus, minus, star, div, opar, cpar, num = G.Terminals('+ - * / ( ) num')
#
# E %= T + X
# X %= plus + T + X | minus + T + X | G.Epsilon
# T %= F + Y
# Y %= star + F + Y | div + F + Y | G.Epsilon
# F %= num | opar + E + cpar

# firsts = compute_firsts(G)

# assert firsts == {
#    plus: ContainerSet(plus , contains_epsilon=False),
#    minus: ContainerSet(minus , contains_epsilon=False),
#    star: ContainerSet(star , contains_epsilon=False),
#    div: ContainerSet(div , contains_epsilon=False),
#    opar: ContainerSet(opar , contains_epsilon=False),
#    cpar: ContainerSet(cpar , contains_epsilon=False),
#    num: ContainerSet(num , contains_epsilon=False),
#    E: ContainerSet(num, opar , contains_epsilon=False),
#    T: ContainerSet(num, opar , contains_epsilon=False),
#    F: ContainerSet(num, opar , contains_epsilon=False),
#    X: ContainerSet(plus, minus , contains_epsilon=True),
#    Y: ContainerSet(div, star , contains_epsilon=True),
#    Sentence(T, X): ContainerSet(num, opar , contains_epsilon=False),
#    Sentence(plus, T, X): ContainerSet(plus , contains_epsilon=False),
#    Sentence(minus, T, X): ContainerSet(minus , contains_epsilon=False),
#    G.Epsilon: ContainerSet( contains_epsilon=True),
#    Sentence(F, Y): ContainerSet(num, opar , contains_epsilon=False),
#    Sentence(star, F, Y): ContainerSet(star , contains_epsilon=False),
#    Sentence(div, F, Y): ContainerSet(div , contains_epsilon=False),
#    Sentence(num): ContainerSet(num , contains_epsilon=False),
#    Sentence(opar, E, cpar): ContainerSet(opar , contains_epsilon=False)
# }
# print(firsts)

# follows = compute_follows(G, firsts)

# print(follows)
#
# assert follows == {
#    E: ContainerSet(G.EOF, cpar , contains_epsilon=False),
#    T: ContainerSet(cpar, plus, G.EOF, minus , contains_epsilon=False),
#    F: ContainerSet(cpar, star, G.EOF, minus, div, plus , contains_epsilon=False),
#    X: ContainerSet(G.EOF, cpar , contains_epsilon=False),
#    Y: ContainerSet(cpar, plus, G.EOF, minus , contains_epsilon=False)
# }

# M = build_parsing_table(G, firsts, follows)

# print(M)
#
# assert M == {
#    ( E, num, ): [ Production(E, Sentence(T, X)), ],
#    ( E, opar, ): [ Production(E, Sentence(T, X)), ],
#    ( X, plus, ): [ Production(X, Sentence(plus, T, X)), ],
#    ( X, minus, ): [ Production(X, Sentence(minus, T, X)), ],
#    ( X, cpar, ): [ Production(X, G.Epsilon), ],
#    ( X, G.EOF, ): [ Production(X, G.Epsilon), ],
#    ( T, num, ): [ Production(T, Sentence(F, Y)), ],
#    ( T, opar, ): [ Production(T, Sentence(F, Y)), ],
#    ( Y, star, ): [ Production(Y, Sentence(star, F, Y)), ],
#    ( Y, div, ): [ Production(Y, Sentence(div, F, Y)), ],
#    ( Y, plus, ): [ Production(Y, G.Epsilon), ],
#    ( Y, G.EOF, ): [ Production(Y, G.Epsilon), ],
#    ( Y, cpar, ): [ Production(Y, G.Epsilon), ],
#    ( Y, minus, ): [ Production(Y, G.Epsilon), ],
#    ( F, num, ): [ Production(F, Sentence(num)), ],
#    ( F, opar, ): [ Production(F, Sentence(opar, E, cpar)), ]
# }

# parser = metodo_predictivo_no_recursivo(G, M)
# left_parse = parser([num, star, num, star, num, plus, num, star, num, plus, num, plus, num, G.EOF])
#
# assert left_parse == [
#    Production(E, Sentence(T, X)),
#    Production(T, Sentence(F, Y)),
#    Production(F, Sentence(num)),
#    Production(Y, Sentence(star, F, Y)),
#    Production(F, Sentence(num)),
#    Production(Y, Sentence(star, F, Y)),
#    Production(F, Sentence(num)),
#    Production(Y, G.Epsilon),
#    Production(X, Sentence(plus, T, X)),
#    Production(T, Sentence(F, Y)),
#    Production(F, Sentence(num)),
#    Production(Y, Sentence(star, F, Y)),
#    Production(F, Sentence(num)),
#    Production(Y, G.Epsilon),
#    Production(X, Sentence(plus, T, X)),
#    Production(T, Sentence(F, Y)),
#    Production(F, Sentence(num)),
#    Production(Y, G.Epsilon),
#    Production(X, Sentence(plus, T, X)),
#    Production(T, Sentence(F, Y)),
#    Production(F, Sentence(num)),
#    Production(Y, G.Epsilon),
#    Production(X, G.Epsilon),
# ]

# endregion


# region test2
#
# G = Grammar()
# S = G.NonTerminal('S', True)
# A,B = G.NonTerminals('A B')
# a,b = G.Terminals('a b')
#
# S %= A + B
# A %= a + A | a
# B %= b + B | b
#
# print(G)
#
# firsts = compute_firsts(G)
# pprint(firsts)
#
# # print(inspect(firsts))
# assert firsts == {
#    a: ContainerSet(a , contains_epsilon=False),
#    b: ContainerSet(b , contains_epsilon=False),
#    S: ContainerSet(a , contains_epsilon=False),
#    A: ContainerSet(a , contains_epsilon=False),
#    B: ContainerSet(b , contains_epsilon=False),
#    Sentence(A, B): ContainerSet(a , contains_epsilon=False),
#    Sentence(a, A): ContainerSet(a , contains_epsilon=False),
#    Sentence(a): ContainerSet(a , contains_epsilon=False),
#    Sentence(b, B): ContainerSet(b , contains_epsilon=False),
#    Sentence(b): ContainerSet(b , contains_epsilon=False)
# }
#
# follows = compute_follows(G, firsts)
# pprint(follows)
#
# # print(inspect(follows))
# assert follows == {
#    S: ContainerSet(G.EOF , contains_epsilon=False),
#    A: ContainerSet(b , contains_epsilon=False),
#    B: ContainerSet(G.EOF , contains_epsilon=False)
# }
#
# M = build_parsing_table(G, firsts, follows)

# endregion


# region test3
#
# G = Grammar()
# S = G.NonTerminal('S', True)
# A,B,C = G.NonTerminals('A B C')
# a,b,c,d,f = G.Terminals('a b c d f')
#
# S %= a + A | B + C | f + B + f
# A %= a + A | G.Epsilon
# B %= b + B | G.Epsilon
# C %= c + C | d
#
# print(G)
#
# firsts = compute_firsts(G)
# pprint(firsts)
#
# # print(inspect(firsts))
# assert firsts == {
#    a: ContainerSet(a , contains_epsilon=False),
#    b: ContainerSet(b , contains_epsilon=False),
#    c: ContainerSet(c , contains_epsilon=False),
#    d: ContainerSet(d , contains_epsilon=False),
#    f: ContainerSet(f , contains_epsilon=False),
#    S: ContainerSet(d, a, f, c, b , contains_epsilon=False),
#    A: ContainerSet(a , contains_epsilon=True),
#    B: ContainerSet(b , contains_epsilon=True),
#    C: ContainerSet(c, d , contains_epsilon=False),
#    Sentence(a, A): ContainerSet(a , contains_epsilon=False),
#    Sentence(B, C): ContainerSet(d, c, b , contains_epsilon=False),
#    Sentence(f, B, f): ContainerSet(f , contains_epsilon=False),
#    G.Epsilon: ContainerSet( contains_epsilon=True),
#    Sentence(b, B): ContainerSet(b , contains_epsilon=False),
#    Sentence(c, C): ContainerSet(c , contains_epsilon=False),
#    Sentence(d): ContainerSet(d , contains_epsilon=False)
# }
#
# follows = compute_follows(G, firsts)
# pprint(follows)
#
# # print(inspect(follows))
# assert follows == {
#    S: ContainerSet(G.EOF , contains_epsilon=False),
#    A: ContainerSet(G.EOF , contains_epsilon=False),
#    B: ContainerSet(d, f, c , contains_epsilon=False),
#    C: ContainerSet(G.EOF , contains_epsilon=False)
# }
#
# M = build_parsing_table(G, firsts, follows)
# pprint(M)
#
# # print(inspect(M))
# assert M == {
#    ( S, a, ): [ Production(S, Sentence(a, A)), ],
#    ( S, c, ): [ Production(S, Sentence(B, C)), ],
#    ( S, b, ): [ Production(S, Sentence(B, C)), ],
#    ( S, d, ): [ Production(S, Sentence(B, C)), ],
#    ( S, f, ): [ Production(S, Sentence(f, B, f)), ],
#    ( A, a, ): [ Production(A, Sentence(a, A)), ],
#    ( A, G.EOF, ): [ Production(A, G.Epsilon), ],
#    ( B, b, ): [ Production(B, Sentence(b, B)), ],
#    ( B, c, ): [ Production(B, G.Epsilon), ],
#    ( B, f, ): [ Production(B, G.Epsilon), ],
#    ( B, d, ): [ Production(B, G.Epsilon), ],
#    ( C, c, ): [ Production(C, Sentence(c, C)), ],
#    ( C, d, ): [ Production(C, Sentence(d)), ]
# }
#
# parser = metodo_predictivo_no_recursivo(G, M)
#
# left_parse = parser([b, b, d, G.EOF])
# pprint(left_parse)
#
# # print(inspect(left_parse))
# assert left_parse == [
#    Production(S, Sentence(B, C)),
#    Production(B, Sentence(b, B)),
#    Production(B, Sentence(b, B)),
#    Production(B, G.Epsilon),
#    Production(C, Sentence(d)),
# ]

# endregion