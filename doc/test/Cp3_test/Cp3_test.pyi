import typing


class Production(object):
    Left: typing.Any
    Right: typing.Any

    def __init__(self, nonTerminal: NonTerminal, sentence: Sentence): ...

    def __str__(self) -> typing.Text: ...

    def __repr__(self) -> typing.Text: ...

    def __iter__(self) -> typing.Iterator[Sentence]: ...

    def __eq__(self, other): ...

    @property
    def IsEpsilon(self) -> typing.Any: ...

class Symbol(object):
    Name: typing.Text
    Grammar: 'Grammar'

    def __init__(self, name: typing.Text, grammar: 'Grammar') -> \
            typing.NoReturn: ...

    def __str__(self) -> Name: ...

    def __repr__(self) -> Name: ...

    def __add__(self, other: 'Symbol') -> 'Sentence(self, other)': ...

    def __or__(self, other: 'Sentence') -> 'SentenceList': ...
    # todo ver como definir usando annotations y el typing los valores del
    #  annotation, no solamente el tipo sino lo que debe tener
    @property
    def IsEpsilon(self) -> False: ...

    def __len__(self) -> 1: ...

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

