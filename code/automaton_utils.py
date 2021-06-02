import sys

import pydot
from grammars_utils import Grammar, is_regular_grammar, Epsilon, build_grammar
from grammars_utils import Item



class NFA:
    def __init__(self, states: int, finals: iter, transitions: dict, start=0):
        '''
        :param states:  representa en número de estados del autómata. Los
        estados se modelarán como números, comenzando en 0 y hasta states-1.
        :type states: int
        :param finals: epresenta la colección de estados finales del autómata.
        Dado que los estados se representan con números, este debe ser una
        colección de números.
        :param transitions: representa la función de transición. Se espera un
        diccionario que, dados como llaves un estado origen (un número) y un
        símbolo (un string), devuelve como valor una colección de estados
        destino (números). Para renotar una ϵϵ-transición usaremos el string
        vacío.
        :param start: estado inicial
        :type start: int
        '''
        self.states = states
        self.start = start
        self.finals = set(finals)
        self.map = transitions
        self.vocabulary = set()
        self.transitions = {state: {} for state in range(states)}

        destinations: list
        origin: int
        symbol: str
        for (origin, symbol), destinations in transitions.items():
            assert hasattr(destinations,
                           '__iter__'), 'Invalid collection of states'
            self.transitions[origin][symbol] = destinations
            self.vocabulary.add(symbol)

        self.vocabulary.discard('')

    def epsilon_transitions(self, state):
        assert state in self.transitions, 'Invalid state'
        try:
            return self.transitions[state]['']
        except KeyError:
            return ()

    def graph(self):
        G = pydot.Dot(rankdir='LR', margin=0.1)
        G.add_node(
            pydot.Node('start', shape='plaintext', label='', width=0, height=0))

        for (start, tran), destinations in self.map.items():
            tran = 'ε' if tran == '' else tran
            G.add_node(pydot.Node(start, shape='circle',
                                  style='bold' if start in self.finals else ''))
            for end in destinations:
                G.add_node(pydot.Node(end, shape='circle',
                                      style='bold' if end in self.finals else ''))
                G.add_edge(pydot.Edge(start, end, label=tran, labeldistance=2))

        G.add_edge(pydot.Edge('start', self.start, label='', style='dashed'))
        return G

    def _repr_svg_(self):
        try:
            return self.graph().create().decode('utf8')
        except:
            pass



class DFA(NFA):

    def __init__(self, states: int, finals: list, transitions: dict, start=0):
        assert all(isinstance(value, int) for value in transitions.values())
        assert all(len(symbol) > 0 for origin, symbol in transitions)

        transitions = {key: [value] for key, value in transitions.items()}
        NFA.__init__(self, states, finals, transitions, start)
        self.current = start

    def epsilon_transitions(self):
        raise TypeError()

    def _move(self, symbol):
        # Your code here
        try:
            self.current = self.transitions[self.current][symbol][0]
        except KeyError: return False

        return True

    def _reset(self):
        self.current = self.start

    def recognize(self, string):
        # Your code here
        self._reset()
        for char in string:
            if not self._move(char):
                return False

        return self.current in self.finals


def reg_grammar2DFA(grammar: Grammar) -> DFA:
    '''
    :param grammar: a regular grammar
    :type grammar: Grammar
    :return: DFA's grammar
    :rtype: DFA
    '''

    count = 0
    states = {}
    # se le asocia a cada no terminal un estado
    for symbol in grammar.nonTerminals:
        states[symbol.Name] = count
        count += 1
    # se verifica que el estado 0 sea el simbolo inicial
    for char in states:
        if states[char] == 0 and char != grammar.startSymbol.Name:
            states[char] = states[grammar.startSymbol.Name]
            states[grammar.startSymbol.Name] = 0
    # se verifica que halla que agregar un estado final si hay producciones
    # que deriven en un no terminal por lo que irian  a ese estado final
    exist_new_final = any([len(prod.Right) == 1 for prod in grammar.Productions])

    finals = []
    final = 0
    if exist_new_final:
        finals.append(count)
        final = 1
    # si el distinguido deriva en e el tambien es final
    for prod in grammar.startSymbol.productions:
        if isinstance(prod.Right, Epsilon):
            finals.append(0)
            break

    transitions = {}
    # se agregan las transiciones
    for prod in grammar.Productions:
        left, right = prod
        if len(right) == 2:
            transitions[(states[left.Name], right[0].Name,)] = states[right[1].Name]

        else:
            if isinstance(right, Epsilon): pass
            else:
                transitions[(states[left.Name], right[0].Name,)] = count

    return DFA(len(states) + final, finals, transitions)


def automaton2reg(automaton: DFA) -> str:
    # se modifican las transiciones sumandole 1 a cada estado para poner el nuevo
    # estado inicial y el nuevo estado final
    transitions = {(state + 1, symbol,): dest[0] + 1 for (state, symbol,), dest in
                   automaton.map.items()}
    transitions[(0, '',)] = 1 # se pone la transicion con el primer estado
    for final in automaton.finals:
        transitions[(final + 1, '', )] = automaton.states + 1
        # se ponen las transiciones con el nuevo ultimo estado
    # se comienza la eliminacion de estados desde el 2 hasta los ultimos del
    # automata no el nuevo estado ultimo que se agrego
    for node in range(2, automaton.states + 1):
        # se buscan las transiciones que el tiene con el mismo, eso es una re
        # de la forma (symbol1|symbol2|...|symbolk)
        inter_reg = '|'.join([symbol for (state, symbol,), dest in
                              transitions.items()
                              if state == node and state == dest])

        new_transitions = {}
        # se buscan todas las transiciones que llegan a el y por cada una todas
        # las que salen de el, se crean las nuevas transiciones con las re que
        # llegan a el con las de el y las que salen de el, notar que de esta
        # forma un estado puede tener mas de una transicion con otro con
        # distintas reg, eso no es problema ya que siempre se buscan en todas las
        # transiciones
        for (start_state, start_reg,), start_dest in transitions.items():
            if start_state != node and start_dest == node:
                for (end_state, end_reg,), end_dest in transitions.items():
                    if end_dest != node and end_state == node:
                        if inter_reg:
                            new_transitions[(start_state,
                                start_reg + f'({inter_reg})*' + end_reg,)] = end_dest
                        else:
                            new_transitions[(start_state,
                                             start_reg + end_reg,)] = end_dest
        # se anaden las transiciones que no incluyen ese estado puesto que ya
        # se elimino
        for (state, reg,), dest in transitions.items():
            if state == node or dest == node: continue
            new_transitions[(state, reg,)] = dest

        transitions = new_transitions
    # por ultimo se elimina el primer estado siguiendo un proceso similar a los
    # otros, primero se crea la re de las re que van de el con el mismo
    inter_reg = '|'.join([reg for (state, reg,), dest in transitions.items()
                         if state == 1 and dest == 1])
    # solo el estado inicial agregado nuevo va a tener transiciones con el, solo
    # una, y el va a tener varias con el nuevo final que se agrego, suiguiendo
    # el mismo procedimiento ahora el nuevo estado inicial agregado tendra varias
    # transiciones con varias re con el nuevo estado final agregado
    new_transitions = {}
    for (start_state, start_reg,), start_dest in transitions.items():
        if start_state != 1 and start_dest == 1:
            for (end_state, end_reg,), end_dest in transitions.items():
                if end_dest != 1 and end_state == 1:
                    if inter_reg:
                        new_transitions[(start_state,
                             start_reg + f'({inter_reg})*' + end_reg,)] = end_dest
                    else:
                        new_transitions[(start_state,
                                         start_reg + end_reg,)] = end_dest
    # se juntan en una solo re aquellas que tenga en paralelo el estado inicial
    # agregado con el final agregado, esa es la resultante
    reg = '|'.join([f'({reg})' for (state, reg,) in new_transitions])
    return reg


class State:
    def __init__(self, state, final=False, formatter=lambda x: str(x)):
        self.state = state
        self.final = final
        self.transitions = {}
        self.epsilon_transitions = set()
        self.tag = None
        self.formatter = formatter

    def set_formatter(self, formatter, visited=None):
        if visited is None:
            visited = set()
        elif self in visited:
            return

        visited.add(self)
        self.formatter = formatter
        for destinations in self.transitions.values():
            for node in destinations:
                node.set_formatter(formatter, visited)
        for node in self.epsilon_transitions:
            node.set_formatter(formatter, visited)
        return self

    def has_transition(self, symbol):
        return symbol in self.transitions

    def add_transition(self, symbol, state):
        try:
            self.transitions[symbol].append(state)
        except:
            self.transitions[symbol] = [state]
        return self

    def add_epsilon_transition(self, state):
        self.epsilon_transitions.add(state)
        return self

    def recognize(self, string):
        states = self.epsilon_closure
        for symbol in string:
            states = self.move_by_state(symbol, *states)
            states = self.epsilon_closure_by_state(*states)
        return any(s.final for s in states)

    def to_deterministic(self, formatter=lambda x: str(x)):
        closure = self.epsilon_closure
        start = State(tuple(closure), any(s.final for s in closure), formatter)

        closures = [closure]
        states = [start]
        pending = [start]

        while pending:
            state = pending.pop()
            symbols = {symbol for s in state.state for symbol in s.transitions}

            for symbol in symbols:
                move = self.move_by_state(symbol, *state.state)
                closure = self.epsilon_closure_by_state(*move)

                if closure not in closures:
                    new_state = State(tuple(closure),
                                      any(s.final for s in closure), formatter)
                    closures.append(closure)
                    states.append(new_state)
                    pending.append(new_state)
                else:
                    index = closures.index(closure)
                    new_state = states[index]

                state.add_transition(symbol, new_state)

        return start

    @staticmethod
    def from_nfa(nfa, get_states=False):
        states = []
        for n in range(nfa.states):
            state = State(n, n in nfa.finals)
            states.append(state)

        for (origin, symbol), destinations in nfa.map.items():
            origin = states[origin]
            origin[symbol] = [states[d] for d in destinations]

        if get_states:
            return states[nfa.start], states
        return states[nfa.start]

    @staticmethod
    def move_by_state(symbol, *states):
        return {s for state in states if state.has_transition(symbol) for s in
                state[symbol]}

    @staticmethod
    def epsilon_closure_by_state(*states):
        closure = {state for state in states}

        l = 0
        while l != len(closure):
            l = len(closure)
            tmp = [s for s in closure]
            for s in tmp:
                for epsilon_state in s.epsilon_transitions:
                    closure.add(epsilon_state)
        return closure

    @property
    def epsilon_closure(self):
        return self.epsilon_closure_by_state(self)

    @property
    def name(self):
        return self.formatter(self.state)

    def get(self, symbol):
        target = self.transitions[symbol]
        assert len(target) == 1
        return target[0]

    def __getitem__(self, symbol):
        if symbol == '':
            return self.epsilon_transitions
        try:
            return self.transitions[symbol]
        except KeyError:
            return None

    def __setitem__(self, symbol, value):
        if symbol == '':
            self.epsilon_transitions = value
        else:
            self.transitions[symbol] = value

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.state)

    def __hash__(self):
        return hash(self.state)

    def __iter__(self):
        yield from self._visit()

    def _visit(self, visited=None) -> 'State':
        if visited is None:
            visited = set()
        elif self in visited:
            return

        visited.add(self)
        yield self

        for destinations in self.transitions.values():
            for node in destinations:
                yield from node._visit(visited)
        for node in self.epsilon_transitions:
            yield from node._visit(visited)

    def graph(self):
        G = pydot.Dot(rankdir='LR', margin=0.1)
        G.add_node(
            pydot.Node('start', shape='plaintext', label='', width=0, height=0))

        visited = set()

        def visit(start):
            ids = id(start)
            if ids not in visited:
                visited.add(ids)
                G.add_node(pydot.Node(ids, label=start.name, shape='circle',
                                      style='bold' if start.final else ''))
                for tran, destinations in start.transitions.items():
                    for end in destinations:
                        visit(end)
                        G.add_edge(pydot.Edge(ids, id(end), label=tran,
                                              labeldistance=2))
                for end in start.epsilon_transitions:
                    visit(end)
                    G.add_edge(
                        pydot.Edge(ids, id(end), label='ε', labeldistance=2))

        visit(self)
        G.add_edge(pydot.Edge('start', id(self), label='', style='dashed'))

        return G

    def _repr_svg_(self):
        try:
            return self.graph().create_svg().decode('utf8')
        except:
            pass

    def write_to(self, fname):
        return self.graph().write_svg(fname)


def multiline_formatter(state):
    return '\n'.join(str(item) for item in state)


def lr0_formatter(state):
    try:
        return '\n'.join(str(item)[:-4] for item in state)
    except TypeError:
        return str(state)[:-4]


if __name__ == '__main__':
    # automaton = NFA(states=3, finals=[2], transitions={
    #     (0, 'a'): [0],
    #     (0, 'b'): [0, 1],
    #     (1, 'a'): [2],
    #     (1, 'b'): [2],
    # })
    #
    # algo = automaton.graph().write_svg(path=r'.\some', prog=r'C:\Program Files (x86)\Graphviz2.38\bin\dot.exe')
    # print(sys.argv[0])
    # g = build_grammar(['S->aS',
# 'S->',
# 'S->aA',
# 'A->bA'])
#     dfa = reg_grammar2DFA(g)
#     algo = dfa.graph().write_svg(path=r'.\some', prog=r'C:\Program Files (x86)\Graphviz2.38\bin\dot.exe')
#     other = DFA(3, [1, 2], {(0, 'a',): 1, (1, 'a',): 1, (1, 'b',): 1,
#                             (0, 'b',): 2, (2, 'a',): 0, (2, 'b',): 2})
#     reg = automaton2reg(dfa)
#     print(reg)


    g = build_grammar(['S->aB', 'S->bS', 'B->bS', 'B->b'])
    dfa = reg_grammar2DFA(g)
    algo = dfa.graph().write_svg(path=r'.\some', prog=r'C:\Program Files (x86)\Graphviz2.38\bin\dot.exe')



