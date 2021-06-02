import json
from typing import List, Union, Dict, Iterable
import pydot



class Production(object):
    '''
    Modelaremos las producciones con la clase Production. Las funcionalidades
    básicas con que contamos son:

    Poder acceder la cabecera (parte izquierda) y cuerpo (parte derecha) de cada
    producción a través de los campos Left y Right respectivamente.
    Consultar si la producción es de la forma X→ϵ a través de la propiedad
    IsEpsilon.
    Desempaquetar la producción en cabecera y cuerpo usando asignaciones:
    left, right = production.

    Las producciones no deben ser instanciadas directamente con la aplicación
    de su constructor.
    '''
    def __init__(self, nonTerminal: 'NonTerminal', sentence: Union['Sentence',
                                                               'SentenceList']):
        self.Left = nonTerminal
        self.Right = sentence

    def __str__(self):
        return '%s := %s' % (self.Left, self.Right)

    def __repr__(self):
        return '%s -> %s' % (self.Left, self.Right)

    def __hash__(self):
        return hash((self.Left, self.Right))

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


class Sentence(object):

    '''
    Modelaremos los oraciones y formas oracionales del lenguaje con la clase
    Sentence. Esta clase funcionará como una colección de terminales y no
    terminales. Entre las funcionalidades básicas que provee tenemos que nos :

    Permite acceder a los símbolos que componen la oración a través del campo
    _symbols de cada instancia.
    Permite conocer si la oración es completamente vacía a través de la
    propiedad IsEpsilon.
    Permite obtener la concatenación con un símbolo u otra oración aplicando el
    operador +.
    Permite conocer la longitud de la oración (cantidad de símbolos que la
    componen) utilizando la función build-in de python len(...).
    '''

    def __init__(self, *args: Union['Terminal', 'NonTerminal']):
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

    def __getitem__(self, index) -> 'Symbol':
        return self._symbols[index]

    def __eq__(self, other):
        return self._symbols == other._symbols

    def __hash__(self):
        return self.hash

    @property
    def IsEpsilon(self):
        return False




class Symbol(object):
    '''
    Símbolos
    Modelaremos los símbolos del lenguaje con la clase Symbol. Esta clase
    funcionará como base para la definición de terminales y no terminales.

    no se debe instanciar a partir del constructor, o sea, no se debe instanciar
    '''
    def __init__(self, name: str, grammar: 'Grammar'):
        '''
        :param name: Podemos conocer si representa la cadena especial epsilon a
        través de la propiedad IsEpsilon que poseen todas las instancias.
        :type name: str

        :param grammar: Podemos acceder a la gramática en la que se definió a
        través del campo Grammar de cada instancia.
        :type grammar: Grammar
        '''
        self.Name = name
        self.Grammar = grammar

    def __str__(self):
        return self.Name

    def __repr__(self):
        return repr(self.Name)

    def __add__(self, other: 'Symbol') -> Sentence:
        '''
        Pueden ser agrupados con el operador + para formar oraciones.
        '''
        if isinstance(other, Symbol):
            return Sentence(self, other)

        raise TypeError(other)

    def __or__(self, other: Sentence) -> 'SentenceList':

        if isinstance(other, (Sentence)):
            return SentenceList(Sentence(self), other)

        raise TypeError(other)

    @property
    def IsEpsilon(self) -> bool:
        '''
        Podemos conocer si representa la cadena especial epsilon a través de la
        propiedad IsEpsilon que poseen todas las instancias.
        :rtype: bool
        '''
        return False

    def __len__(self):
        return 1


class NonTerminal(Symbol):
    '''
    no se debe instanciar

    Los símbolos no terminales los modelaremos con la clase NonTerminal. Dicha
    clase extiende la clase Symbol para:

    Añadir noción de las producción que tiene al no terminal como cabecera.
    Estas pueden ser conocidas a través del campo productions de cada instancia.
    Permitir añadir producciones para ese no terminal a través del operador %=.
    Incluir propiedades IsNonTerminal - IsTerminal que devolveran True - False
    respectivamente.

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
    '''
    no se debe instanciar
    '''
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
    '''
    Modelaremos el símbolo de fin de cadena con la clase EOF. Dicha clase
    extiende la clases Terminal para heradar su comportamiento.

    La clase EOF no deberá ser instanciada directamente con la aplicación de su
    constructor. En su lugar, una instancia concreta para determinada gramática
    G de Grammar se construirá automáticamente y será accesible a través de
    G.EOF.
    '''

    def __init__(self, Grammar):
        super().__init__('$', Grammar)


class SentenceList(object):
    '''
    Las oraciones pueden ser agrupadas usando el operador |. Esto nos será
    conveniente para definir las producciones que tengan la
    misma cabeza (no terminal en la parte izquierda) en una única sentencia. El
    grupo de oraciones se maneja con la clase SentenceList.
    '''

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
    '''
    Modelaremos tanto la cadena vacía como el símbolo que la representa: epsilon
    (ϵϵ), en la misma clase: Epsilon. Dicha clase extiende las clases Terminal y
    Sentence por lo que ser comporta como ambas. Sobreescribe la implementación
    del método IsEpsilon para indicar que en efecto toda instancia de la clase
    reprensenta epsilon.

    La clase Epsilon no deberá ser instanciada directamente con la aplicación
    de su constructor.
    '''
    def __init__(self, grammar: 'Grammar'):
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

        self.Productions: List[Production] = []
        self.nonTerminals: List[NonTerminal] = []
        self.terminals: List[Terminal] = []
        self.startSymbol: NonTerminal = None
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

        assert type(production) == self.pType, "The Productions most be of only " \
                                               "1 type."

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

    def AugmentedGrammar(self, force=False):
        if not self.IsAugmentedGrammar or force:

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

    def extend(self, values):
        change = False
        for value in values:
            change |= self.add(value)
        return change

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



class Item:

    def __init__(self, production, pos, lookaheads=[]):
        self.production = production
        self.pos = pos
        self.lookaheads = tuple(look for look in lookaheads)

    def __str__(self):
        s = str(self.production.Left) + " -> "
        if len(self.production.Right) > 0:
            for i,c in enumerate(self.production.Right):
                if i == self.pos:
                    s += "."
                s += str(self.production.Right[i])
            if self.pos == len(self.production.Right):
                s += "."
        else:
            s += "."
        s += ", " + str(self.lookaheads)
        return s

    def __repr__(self):
        return str(self)


    def __eq__(self, other):
        return (
            (self.pos == other.pos) and
            (self.production == other.production) and
            (self.lookaheads == other.lookaheads)
        )

    def __hash__(self):
        return hash((self.production,self.pos,self.lookaheads))

    @property
    def IsReduceItem(self):
        return len(self.production.Right) == self.pos

    @property
    def NextSymbol(self):
        if self.pos < len(self.production.Right):
            return self.production.Right[self.pos]
        else:
            return None

    def NextItem(self):
        if self.pos < len(self.production.Right):
            return Item(self.production,self.pos+1,self.lookaheads)
        else:
            return None

    def Preview(self, skip=1):
        unseen = self.production.Right[self.pos+skip:]
        return [ unseen + (lookahead,) for lookahead in self.lookaheads ]

    def Center(self):
        return Item(self.production, self.pos)


class DerivationTree:

    def __init__(self, value: Symbol, parent: 'DerivationTree'=None):
        self.sons = []
        self.parent = parent
        self.value = value

    def add_son(self, son: 'DerivationTree'):
        self.sons.append(son)

    def print_tree(self):
        print(self.value.Name)
        print()
        for son in self.sons:
            son.print_tree()

    def graph(self):
        G = pydot.Dot(rankdir='LR', margin=0.1)
        self.count = 1
        count = 1
        G.add_node(
            pydot.Node(self.value.Name + str(self.count), shape='circle', label=self.value.Name, width=0, height=0))
        stack = [self]
        while stack:
            actual_node = stack.pop()

            for node in actual_node.sons:
                count += 1
                node.count = count
                stack.append(node)

            for node in actual_node.sons:
                G.add_node(pydot.Node(node.value.Name + str(node.count),
                                      shape='circle', label=node.value.Name))
                G.add_edge(pydot.Edge(actual_node.value.Name +
                                      str(actual_node.count), node.value.Name + str(node.count)))
        # for (start, tran), destinations in self.map.items():
        #     tran = 'ε' if tran == '' else tran
        #     G.add_node(pydot.Node(start, shape='circle',
        #                           style='bold' if start in self.finals else ''))
        #     for end in destinations:
        #         G.add_node(pydot.Node(end, shape='circle',
        #                               style='bold' if end in self.finals else ''))
        #         G.add_edge(pydot.Edge(start, end, label=tran, labeldistance=2))
        #
        # G.add_edge(pydot.Edge('start', self.start, label='', style='dashed'))
        return G


def is_regular_grammar(grammar: Grammar):
    for production in grammar.Productions:
        left, rigth = production
        if len(rigth) > 2: return False
        if isinstance(rigth, Epsilon) and left != grammar.startSymbol:
            return False
        if len(rigth) == 1 and not isinstance(rigth, Epsilon) \
                and not isinstance(rigth[0], Terminal): return False
        if len(rigth) == 2 and (not isinstance(rigth[0], Terminal) or
            not isinstance(rigth[1], NonTerminal)): return False

    return True


class Node:

    def __init__(self, symbol: Symbol, number: int, eos=False):
        self.value = symbol.Name
        self.symbol = symbol
        self.sons: Dict[str, 'Node'] = {}
        self.number = number
        self.is_oes = eos


class Tree:

    def __init__(self):
        self.root = Node(Symbol('$', None), 1)

    def insert(self, sentence: Sentence):
        if isinstance(sentence, Epsilon): return
        actual_node = self.root
        for symbol in sentence:
            if symbol.Name not in actual_node.sons:
                actual_node.sons[symbol.Name] = Node(symbol, 1)

            else:
                actual_node.sons[symbol.Name].number += 1
            actual_node = actual_node.sons[symbol.Name]

        actual_node.is_oes = True

    def common_prefix(self) -> str:
        actual_node = self.root
        number = 1
        prefix = ''
        change = True

        for node in self.root.sons.values():
            if node.number > number:
                number = node.number
                actual_node = node
                prefix += node.value
                change = False
                break

        while not change:
            change = True

            for node in actual_node.sons.values():
                if node.number == number:
                    prefix += node.value
                    actual_node = node
                    change = False
                    break

        return prefix


def factorize_grammar(grammar: Grammar) -> Grammar:
    count = 1
    change = True
    productions = [prod for prod in grammar.Productions]

    while change:
        change = False
        new_productions = []
        for noterminal in set([prod.Left for prod in productions]):
            tree = Tree()
            for prod in [prod for prod in productions if prod.Left == noterminal]:
                tree.insert(prod.Right)

            common_prefix = tree.common_prefix()
            if not common_prefix:
                new_productions.extend([prod for prod in productions if prod.Left == noterminal])
                continue

            change = True
            new_noterm = NonTerminal(f'new_noterm{count}', None)
            count += 1

            for prod in  [prod for prod in productions if prod.Left == noterminal]:
                if is_prefix(common_prefix, prod.Right):
                    prod_rest = prod.Right[len(common_prefix):]

                    if not prod_rest:
                        new_prod = Production(new_noterm, Epsilon(None))
                    else: new_prod = Production(new_noterm, Sentence(*prod_rest))
                    new_productions.append(new_prod)
                else:
                    new_productions.append(prod)

            sentence = []
            for char in common_prefix:
                if not char.isupper():
                    sentence.append(Terminal(char, None))
                else: sentence.append(NonTerminal(char, None))
            sentence.append(new_noterm)

            new_productions.append(Production(noterminal, Sentence(*sentence)))

        productions = new_productions

    g = Grammar()
    terminals = set()
    noterminals = set()
    for prod in productions:
        left, rigth = prod
        noterminals.add(left.Name)
        if isinstance(rigth, Epsilon):
            continue
        for symbol in rigth:
            if isinstance(symbol, NonTerminal):
                noterminals.add(symbol.Name)
            else: terminals.add(symbol.Name)

    g.Terminals(' '.join(terminals))
    g.NonTerminals(' '.join(noterminals))
    g.startSymbol = g.symbDict[grammar.startSymbol.Name]

    for prod in productions:
        left, rigth = prod
        if isinstance(rigth, Epsilon):
            g.Add_Production(Production(g.symbDict[left.Name], g.Epsilon))
        else:
            sentence = []
            for symbol in rigth:
                sentence.append(g.symbDict[symbol.Name])
            g.Add_Production(Production(g.symbDict[left.Name], Sentence(*sentence)))

    return g


def delete_unit_production(grammar: Grammar) -> Grammar:
    reemplazable: Dict[NonTerminal, set]
    reemplazable = {noter.Left: set() for noter in grammar.Productions}
    for pro in grammar.Productions:
        if len(pro.Right) == 1 and not isinstance(pro.Right, Epsilon) and \
                isinstance(pro.Right[0], NonTerminal):
            reemplazable[pro.Left].add(pro.Right[0])

    change = True
    while change:
        change = False

        for noter, Inoter in reemplazable.items():
            new = set()
            Inoter_len = len(Inoter)
            for reemp in Inoter:
                new.update(pro.Right[0] for pro in grammar.Productions
                           if pro.Left == reemp and not isinstance(pro.Right, Epsilon)
                           and len(pro.Right) == 1 and isinstance(pro.Right[0], NonTerminal))
            reemplazable[noter].update(new)
            change = (len(reemplazable[noter]) != Inoter_len) or change

    new_pro = [pro for pro in grammar.Productions if len(pro.Right) > 1 or
               (not isinstance(pro.Right, Epsilon) and isinstance(pro.Right[0], Terminal))]
    new_set = []
    for noter, Inoter in reemplazable.items():
        for reemp in Inoter:
            new_set.extend(Production(noter, pro.Right)
                for pro in grammar.Productions if pro.Left == reemp
                and len(pro.Right) > 1 or (not isinstance(pro.Right, Epsilon) and
                isinstance(pro.Right[0], Terminal)))
    new_pro.extend(new_set)

    return build_grammar_from_pro(grammar.startSymbol, new_pro)


def intermediate_fnc(grammar: Grammar) -> Grammar:
    productions = [prod for prod in grammar.Productions if not
        isinstance(prod.Right, Epsilon)]

    new_prod = {pro.Right[0].Name: pro.Left for pro in grammar.Productions
                if len(pro.Right) == 1 and not isinstance(pro.Right, Epsilon)
                and isinstance(pro.Right[0], Terminal)}
    new_productions = []
    noterm = 'noterm'
    count = 1
    for prod in productions:
        if len(prod.Right) == 1: continue
        sentence = []
        for symbol in prod.Right:
            if isinstance(symbol, Terminal):
                if symbol.Name not in new_prod:
                    noterminal = NonTerminal(noterm + str(count), None)
                    new_prod[symbol.Name] = noterminal
                    count += 1
                    new_productions.append(Production(noterminal, Sentence(symbol)))
                    sentence.append(noterminal)
                else:
                    sentence.append(new_prod[symbol.Name])

            else: sentence.append(symbol)

        new_productions.append(Production(prod.Left, Sentence(*sentence)))

    g = Grammar()
    # se construye la gramatica
    # region
    noterminals = set()
    terminals = set()
    for prod in new_productions:
        noterminals.add(prod.Left.Name)
        if isinstance(prod.Right, Epsilon): continue
        for symbol in prod.Right:
            if isinstance(symbol, NonTerminal):
                noterminals.add(symbol.Name)
            else: terminals.add(symbol.Name)

    g.NonTerminals(' '.join(noterminals))
    g.Terminals(' '.join(terminals))

    for prod in new_productions:
        if isinstance(prod.Right, Epsilon):
            g.Add_Production(Production(g.symbDict[prod.Left.Name], g.Epsilon))
        else:
            sentence = []
            for symbol in prod.Right:
                sentence.append(g.symbDict[symbol.Name])
            g.Add_Production(Production(g.symbDict[prod.Left.Name],
                                        Sentence(*sentence)))
    g.startSymbol = g.symbDict[grammar.startSymbol.Name]
    # endregion

    return g


def glc2fnc(grammar: Grammar) -> Grammar:
    g1 = delete_unit_production(grammar)
    new_grammar = intermediate_fnc(g1)
    productions = [prod for prod in new_grammar.Productions]
    new_name = 'new_symbol'
    count = 1

    change = True
    while change:
        change = False
        new_productions = []
        for prod in productions:
            if len(prod.Right) > 2:
                new_symbol = NonTerminal(new_name + str(count), None)
                new_prod = Production(new_symbol, Sentence(*prod.Right[1:]))
                redo_prod = Production(prod.Left, Sentence(prod.Right[0], new_symbol))
                new_productions.append(new_prod)
                new_productions.append(redo_prod)
                change = True
            else: new_productions.append(prod)

        productions = new_productions

    g = Grammar()
    # se construye la gramatica
    # region
    noterminals = set()
    terminals = set()
    for prod in productions:
        noterminals.add(prod.Left.Name)
        if isinstance(prod.Right, Epsilon): continue
        for symbol in prod.Right:
            if isinstance(symbol, NonTerminal):
                noterminals.add(symbol.Name)
            else:
                terminals.add(symbol.Name)

    g.NonTerminals(' '.join(noterminals))
    g.Terminals(' '.join(terminals))

    for prod in productions:
        if isinstance(prod.Right, Epsilon):
            g.Add_Production(Production(g.symbDict[prod.Left.Name], g.Epsilon))
        else:
            sentence = []
            for symbol in prod.Right:
                sentence.append(g.symbDict[symbol.Name])
            g.Add_Production(Production(g.symbDict[prod.Left.Name],
                                        Sentence(*sentence)))
    g.startSymbol = g.symbDict[grammar.startSymbol.Name]
    # endregion

    return g


def build_grammar_from_pro(start_symbol: NonTerminal,
                           productions: Iterable[Production]) -> Grammar:
    g = Grammar()

    noterminals = set()
    terminals = set()
    for prod in productions:
        noterminals.add(prod.Left.Name)
        if isinstance(prod.Right, Epsilon): continue
        for symbol in prod.Right:
            if isinstance(symbol, NonTerminal):
                noterminals.add(symbol.Name)
            else:
                terminals.add(symbol.Name)

    g.NonTerminals(' '.join(noterminals))
    g.Terminals(' '.join(terminals))

    for prod in productions:
        if isinstance(prod.Right, Epsilon):
            g.Add_Production(Production(g.symbDict[prod.Left.Name], g.Epsilon))
        else:
            sentence = []
            for symbol in prod.Right:
                sentence.append(g.symbDict[symbol.Name])
            g.Add_Production(Production(g.symbDict[prod.Left.Name],
                                        Sentence(*sentence)))
    g.startSymbol = g.symbDict[start_symbol.Name]

    return g


def is_prefix(prefix: str, sentence: Sentence):
    if len(prefix) > len(sentence): return False
    for index in range(len(prefix)):
        if prefix[index] != sentence[index].Name: return False

    return True


def delete_left_recursivity(grammar: Grammar) -> Grammar:
    new_productions = []
    count = 1
    for noterminal in grammar.nonTerminals:
        left_recursion = [prod for prod in noterminal.productions
            if not isinstance(prod.Right, Epsilon) and
                          prod.Right[0] == noterminal]
        no_recursion = [prod for prod in noterminal.productions
            if not isinstance(prod.Right, Epsilon) and
                        prod.Right[0] != noterminal]
        if left_recursion:
            new_noter = NonTerminal(f'name{count}', None)
            for prod in no_recursion:
                rigth = [el for el in prod.Right]
                rigth.append(new_noter)
                redo_prod = Production(noterminal, Sentence(*rigth))
                new_productions.append(redo_prod)
            for prod in left_recursion:
                rigth = []
                rigth.extend(prod.Right[1:])
                rigth.append(new_noter)
                new_prod = Production(new_noter, Sentence(*rigth))
                new_productions.append(new_prod)
            count += 1
            new_productions.append(Production(new_noter, Epsilon(None)))
        else: new_productions.extend(noterminal.productions)

    g = Grammar()
    # se construye la gramatica
    # region
    noterminals = set()
    terminals = set()
    for prod in new_productions:
        noterminals.add(prod.Left.Name)
        if isinstance(prod.Right, Epsilon): continue
        for symbol in prod.Right:
            if isinstance(symbol, NonTerminal):
                noterminals.add(symbol.Name)
            else:
                terminals.add(symbol.Name)

    g.NonTerminals(' '.join(noterminals))
    g.Terminals(' '.join(terminals))

    for prod in new_productions:
        if isinstance(prod.Right, Epsilon):
            g.Add_Production(Production(g.symbDict[prod.Left.Name], g.Epsilon))
        else:
            sentence = []
            for symbol in prod.Right:
                sentence.append(g.symbDict[symbol.Name])
            g.Add_Production(Production(g.symbDict[prod.Left.Name],
                                        Sentence(*sentence)))
    g.startSymbol = g.symbDict[grammar.startSymbol.Name]
    # endregion

    return g

def build_grammar(productions: List[str]):
    g = Grammar()
    terminals = set()
    noterminals = set()
    for prod in productions:
        left, right = prod.split('->')
        noterminals.add(left)
        for symbol in right:
            if not symbol.isupper():
                terminals.add(symbol)
            else: noterminals.add(symbol)

    g.NonTerminals(' '.join(noterminals))
    g.Terminals(' '.join(terminals))

    for prod in productions:
        left, right = prod.split('->')
        sentence = []
        for symbol in right:
            sentence.append(g.symbDict[symbol])
        if sentence:
            g.Add_Production(Production(g.symbDict[left], Sentence(*sentence)))
        else:
            g.Add_Production(Production(g.symbDict[left], g.Epsilon))

    g.startSymbol = g.symbDict[productions[0].split('->')[0]]
    return g


def modify_tablell1(table):
    new_table = {}
    for (noterminal, symbol), action in table.items():
        new_table[noterminal, str(symbol)] = action

    return new_table



if __name__ == '__main__':
    # g = build_grammar(['S->aS', 'S->bA', 'S->', 'A->aS'])
    #
    # # print(is_regular_grammar(g))
    #
    # g = build_grammar(['S->caAB', 'S->caABT', 'S->caAK', 'S->bcT',
    #                    'S->bZ', 'S->', 'A->t', 'B->q', 'T->as', 'K->ds',
    #                    'Z->f'])
    #
    # t = factorize_grammar(g)
    # print(t)
    #
    # g = build_grammar(['S->aX', 'S->bY', 'X->Ya', 'X->ba', 'Y->bXX',
    #                   'Y->aba'])
    #
    # t = intermediate_fnc(g)
    # t = glc2fnc(g)
    # print(t)

    g = build_grammar(['E->E+T',
'E->T',
'T->T*F',
'T->F',
'F->a',
'F->(E)'])

    g1 = delete_unit_production(g)
    g2 = glc2fnc(g1)
    print(g2)
    pass


