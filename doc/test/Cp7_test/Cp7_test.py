from Cp7_test.cmp.tools import NFA, DFA
from Cp7_test.cmp.utils import DisjointSet, DisjointNode


def automata_union(a1: NFA, a2: NFA) -> NFA:
    transitions = {}

    start = 0
    d1 = 1  # cantidad de nuevos estados para redefinir los estados de a1
    d2 = a1.states + d1  # se redefinen los nombres de los estados del automata
    # 2 usando la cantidad de estados de a1
    final = a2.states + d2

    for (origin, symbol), destinations in a1.map.items():
        ## Relocate a1 transitions ...
        # Your code here
        transitions[origin + d1, symbol] = [x + d1 for x in destinations]

    for (origin, symbol), destinations in a2.map.items():
        ## Relocate a2 transitions ...
        # Your code here
        transitions[origin + d2, symbol] = [x + d2 for x in destinations]

    ## Add transitions from start state ...
    # Your code here
    transitions[start, ''] = [a1.start + d1, a2.start + d2]
    ## Add transitions to final state ...
    # Your code here

    for ancent_final in a1.finals:
        transitions[ancent_final + d1, ''] = [final]

    for ancent_final in a2.finals:
        transitions[ancent_final + d2, ''] = [final]

    # ..................................................
    states = a1.states + a2.states + 2
    finals = {final}

    return NFA(states, finals, transitions, start)


def automata_concatenation(a1, a2):
    transitions = {}

    start = 0
    d1 = 0
    d2 = a1.states + d1
    final = a2.states + d2

    for (origin, symbol), destinations in a1.map.items():
        ## Relocate a1 transitions ...
        # Your code here
        transitions[origin, symbol] = [x for x in destinations]

    for (origin, symbol), destinations in a2.map.items():
        ## Relocate a2 transitions ...
        # Your code here
        transitions[origin + d2, symbol] = [x + d2 for x in destinations]

    ## Add transitions from start state ...
    # Your code here
    for ancent_final in a1.finals:
        transitions[ancent_final, ''] = [a2.start + d2]

    ## Add transitions to final state ...
    # Your code here
    for ancent_final in a2.finals:
        transitions[ancent_final + d2, ''] = [final]

    states = a1.states + a2.states + 1
    finals = {final}

    return NFA(states, finals, transitions, start)


def automata_closure(a1: NFA) -> NFA:
    transitions = {}

    start = 0
    d1 = 1
    final = a1.states + d1

    for (origin, symbol), destinations in a1.map.items():
    ## Relocate automaton transitions ...
    # Your code here
        transitions[origin + d1, symbol] = [x + d1 for x in destinations]

    ## Add transitions from start state ...
    # Your code here
    transitions[start, ''] = [a1.start + d1, final]

    ## Add transitions to final state and to start state ...
    # Your code here
    for ancent_final in a1.finals:
        transitions[ancent_final + d1, ''] = [final]

    transitions[final, ''] = [start]

    states = a1.states + 2
    finals = {final}

    return NFA(states, finals, transitions, start)


# automaton = DFA(states=2, finals=[1], transitions={
#     (0,'a'):  0,
#     (0,'b'):  1,
#     (1,'a'):  0,
#     (1,'b'):  1,
# })

#union = automata_union(automaton, automaton)
#display(union)
# recognize = nfa_to_dfa(union).recognize
#
# assert union.states == 2 * automaton.states + 2
#
# assert recognize('b')
# assert recognize('abbb')
# assert recognize('abaaababab')
#
# assert not recognize('')
# assert not recognize('a')
# assert not recognize('abbbbaa')

# concat = automata_concatenation(automaton, automaton)
#display(concat)
# recognize = nfa_to_dfa(concat).recognize
#
# assert concat.states == 2 * automaton.states + 1
#
# assert recognize('bb')
# assert recognize('abbb')
# assert recognize('abaaababab')
#
# assert not recognize('')
# assert not recognize('a')
# assert not recognize('b')
# assert not recognize('ab')
# assert not recognize('aaaab')
# assert not recognize('abbbbaa')

# closure = automata_closure(automaton)
# #display(closure)
# recognize = nfa_to_dfa(closure).recognize
#
# assert closure.states == automaton.states + 2
#
# assert recognize('')
# assert recognize('b')
# assert recognize('ab')
# assert recognize('bb')
# assert recognize('abbb')
# assert recognize('abaaababab')
#
# assert not recognize('a')
# assert not recognize('abbbbaa')


def distinguish_states(group, automaton: DFA, partition: DisjointSet):
    """

    :param group: lista de disjointNodes
    :param automaton:
    :param partition:
    :return:
    """
    split = set()
    vocabulary = tuple(automaton.vocabulary)
    # me enrede demasiado con los types y con el set, a qien se le va a ocurrir
    # q no se pueden crear conjuntos de conjuntos
    member: DisjointNode
    for member in group:
        # Your code here
        another_member: DisjointNode
        for another_member in group:
            if member != another_member:
                for symbol in vocabulary:
                    try:
                        destiny_member = \
                            automaton.transitions[member.value][symbol][0]
                    except KeyError: destiny_member = False

                    try:
                        destiny_another_member = \
                            automaton.transitions[another_member.value][symbol][0]
                    except KeyError: destiny_another_member = False

                    # pq el 0 evalua como falso aunq sea entero
                    if ((isinstance(destiny_member, int) and not
                        isinstance(destiny_another_member, int)) or (
                            not isinstance(destiny_member, int) and
                            isinstance(destiny_another_member, int))):
                        split.add(another_member.value)
                        break

                    if ((isinstance(destiny_member, int) and
                        isinstance(destiny_another_member, int)) and
                            partition[destiny_member].representative !=
                            partition[destiny_another_member].representative):
                        split.add(another_member.value)
                        break

        if len(split) > 0: break

    if len(split) > 0:
        split = [{state for state in split},
                 {node.value for node in group if node.value not in split}]

    else: split = [{node.value for node in group}]

    return split


def state_minimization(automaton):
    partition = DisjointSet(*range(automaton.states))

    ## partition = { NON-FINALS | FINALS }
    partition.merge([state for state in automaton.finals])
    partition.merge([state for state in range(automaton.states) if state not in
                     automaton.finals])
    # Your code here

    while True:
        new_partition = DisjointSet(*range(automaton.states))

        ## Split each group if needed (use distinguish_states(group, automaton,
        # partition))
        new_groups = []
        for group in partition.groups:
            if len(group) > 1:
                new_groups.extend(distinguish_states(group, automaton,
                                                        partition))

        for group in new_groups:
            new_partition.merge(group)
        # Your code here

        if len(new_partition) == len(partition):
            break

        partition = new_partition

    return partition


def automata_minimization(automaton):
    partition = state_minimization(automaton)

    states = [s for s in partition.representatives]

    transitions = {}
    state: DisjointNode
    for i, state in enumerate(states):
        origin = state.value
        state.index = i
        # Your code here
        for symbol, destinations in automaton.transitions[origin].items():
            # Your code here
            representative = \
                partition[
                    automaton.transitions[origin][symbol][0]].representative

            try:
                transitions[i, symbol]
                assert False
            except KeyError:
                # Your code here
                transitions[i, symbol] = representative

    transitions = {(i, symbol): representative.index
                   for (i, symbol), representative in transitions.items()}

    finals = {partition[final].representative.index for final in automaton.finals}
    start = partition[automaton.start].representative.index
    # Your code here

    return DFA(len(states), finals, transitions, start)


# dset = DisjointSet(*range(10))
# print('> Inicializando conjuntos disjuntos:\n', dset)
#
# dset.merge([5,9])
# print('> Mezclando conjuntos 5 y 9:\n', dset)
#
# dset.merge([8,0,2])
# print('> Mezclando conjuntos 8, 0 y 2:\n', dset)
#
# dset.merge([2,9])
# print('> Mezclando conjuntos 2 y 9:\n', dset)
#
# print('> Representantes:\n', dset.representatives)
# print('> Grupos:\n', dset.groups)
# print('> Conjunto 0:\n', dset[0], '--->', type(dset[0]))
# print('> Conjunto 0 [valor]:\n', dset[0].value, '--->' , type(dset[0].value))
# print('> Conjunto 0 [representante]:\n', dset[0].representative, '--->' ,
#       type(dset[0].representative))


automaton = DFA(states=5, finals=[4], transitions={
    (0,'a'): 1,
    (0,'b'): 2,
    (1,'a'): 1,
    (1,'b'): 3,
    (2,'a'): 1,
    (2,'b'): 2,
    (3,'a'): 1,
    (3,'b'): 4,
    (4,'a'): 1,
    (4,'b'): 2,
})


# states = state_minimization(automaton)
# print(states)
#
# for members in states.groups:
#     all_in_finals = all(m.value in automaton.finals for m in members)
#     none_in_finals = all(m.value not in automaton.finals for m in members)
#     assert all_in_finals or none_in_finals
#
# assert len(states) == 4
# assert states[0].representative == states[2].representative
# assert states[1].representative == states[1]
# assert states[3].representative == states[3]
# assert states[4].representative == states[4]

mini = automata_minimization(automaton)
#display(mini)

assert mini.states == 4

assert mini.recognize('abb')
assert mini.recognize('ababbaabb')

assert not mini.recognize('')
assert not mini.recognize('ab')
assert not mini.recognize('aaaaa')
assert not mini.recognize('bbbbb')
assert not mini.recognize('abbabababa')