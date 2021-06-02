import pydot
from Cp7_test.cmp.utils import ContainerSet


class NFA:
    def __init__(self, states: int, finals: iter, transitions: dict, start=0):
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


# region string recognize by automaton test

# automaton = DFA(states=3, finals=[2], transitions={
#     (0, 'a'): 0,
#     (0, 'b'): 1,
#     (1, 'a'): 2,
#     (1, 'b'): 1,
#     (2, 'a'): 0,
#     (2, 'b'): 1,
# })
#
# assert automaton.recognize('ba')
# assert automaton.recognize('aababbaba')
#
# assert not automaton.recognize('')
# assert not automaton.recognize('aabaa')
# assert not automaton.recognize('aababb')

# endregion


def move(automaton: NFA, states: iter, symbol: str):
    moves = set()
    # e-closure: es para un estado o cjto de estados aqiellos estados a los que
    # puedo llehar con  e
    # move: cto de estados a los que puedo llegar partiendo de un estado con un
    # simbolo determinado, e- closure puede ser tipo bfs, o sea con e desde un
    # estado y para esos a los q se llegue con e tambien
    # todo a los estados habria q calcularle la clausura y sobe la misma aplicar
    #  move, el resultado tambien formaria parte del move original, ya q a la
    #  clausura se puede llegar sin consumir nada y a partir de ellos podemos
    #  consumir la letra llegando a mas estados
    for state in states:
        # Your code here
        try:
            moves.update(automaton.transitions[state][symbol])
        except: pass

    return moves


def epsilon_closure(automaton, states: iter):
    pending = [s for s in states]
    closure = {s for s in states}

    while pending:
        state = pending.pop()
        # Your code here
        # todo no me gusta este codigo
        for q in automaton.epsilon_transitions(state):
            if not q in closure:  # todo aqi deberia ser en la clausura
                pending.append(q)
                closure.add(q)

    return ContainerSet(*closure)


def nfa_to_dfa(automaton):
    transitions = {}

    start = epsilon_closure(automaton, [automaton.start])
    start.id = 0
    start.is_final = any(s in automaton.finals for s in start)
    states = [start]
    next_id = 1

    pending = [start]
    while pending:
        state = pending.pop()

        for symbol in automaton.vocabulary:
            # Your code here
            next_state = ContainerSet(*move(automaton, state, symbol))

            next_state.update(epsilon_closure(automaton, next_state))

            if len(next_state) > 0:
                if next_state not in states:
                    next_state.id = next_id
                    next_id += 1

                    next_state.is_final = any(s in automaton.finals for s in
                                              next_state)

                    states.append(next_state)
                    pending.append(next_state)

                else:
                    try:
                        next_state = states[states.index(next_state)]
                    except: raise
                # .............................

                try:
                    transitions[state.id, symbol]
                    assert False, 'Invalid DFA!!!'
                except KeyError:
                    # Your code here
                    transitions[state.id, symbol] = next_state.id


    finals = [state.id for state in states if state.is_final]
    dfa = DFA(len(states), finals, transitions)
    return dfa


automaton = NFA(states=6, finals=[3, 5], transitions={
    (0, ''): [ 1, 2 ],
    (1, ''): [ 3 ],
    (1,'b'): [ 4 ],
    (2,'a'): [ 4 ],
    (3,'c'): [ 3 ],
    (4, ''): [ 5 ],
    (5,'d'): [ 5 ]
})

# region move test

# assert move(automaton, [1], 'a') == set()
# assert move(automaton, [2], 'a') == {4}
# assert move(automaton, [1, 5], 'd') == {5}

# endregion

# region e-cloure test
#
# assert epsilon_closure(automaton, [0]) == {0,1,2,3}
# assert epsilon_closure(automaton, [0, 4]) == {0,1,2,3,4,5}
# assert epsilon_closure(automaton, [1, 2, 4]) == {1,2,3,4,5}

# endregion

dfa = nfa_to_dfa(automaton)
# display(dfa)

assert dfa.states == 4
assert len(dfa.finals) == 4

assert dfa.recognize('')
assert dfa.recognize('a')
assert dfa.recognize('b')
assert dfa.recognize('cccccc')
assert dfa.recognize('adddd')
assert dfa.recognize('bdddd')

assert not dfa.recognize('dddddd')
assert not dfa.recognize('cdddd')
assert not dfa.recognize('aa')
assert not dfa.recognize('ab')
assert not dfa.recognize('ddddc')


automaton = NFA(states=3, finals=[2], transitions={
    (0,'a'): [ 0 ],
    (0,'b'): [ 0, 1 ],
    (1,'a'): [ 2 ],
    (1,'b'): [ 2 ],
})
#display(automaton)

print("No sé que lenguaje reconoce :'(")


assert move(automaton, [0, 1], 'a') == {0, 2}
assert move(automaton, [0, 1], 'b') == {0, 1, 2}

dfa = nfa_to_dfa(automaton)
#display(dfa)

assert dfa.states == 4
assert len(dfa.finals) == 2

assert dfa.recognize('aba')
assert dfa.recognize('bb')
assert dfa.recognize('aaaaaaaaaaaba')

assert not dfa.recognize('aaa')
assert not dfa.recognize('ab')
assert not dfa.recognize('b')
assert not dfa.recognize('')


automaton = NFA(states=5, finals=[4], transitions={
    (0,'a'): [ 0, 1 ],
    (0,'b'): [ 0, 2 ],
    (0,'c'): [ 0, 3 ],
    (1,'a'): [ 1, 4 ],
    (1,'b'): [ 1 ],
    (1,'c'): [ 1 ],
    (2,'a'): [ 2 ],
    (2,'b'): [ 2, 4 ],
    (2,'c'): [ 2 ],
    (3,'a'): [ 3 ],
    (3,'b'): [ 3 ],
    (3,'c'): [ 3, 4 ],
})
#display(automaton)

print("No sé que lenguaje reconoce :'(")

dfa = nfa_to_dfa(automaton)
#display(dfa)

assert dfa.states == 15
assert len(dfa.finals) == 7

assert dfa.recognize('abccac')
assert dfa.recognize('bbbbbbbbaa')
assert dfa.recognize('cac')

assert not dfa.recognize('abbbbc')
assert not dfa.recognize('a')
assert not dfa.recognize('')
assert not dfa.recognize('acacacaccab')