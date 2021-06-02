from typing import Tuple, List

import automaton_utils
import grammars_utils
from automaton_utils import State, multiline_formatter
from grammars_utils import Item, ContainerSet
from first_follow import compute_local_first, compute_firsts


def build_LR0_automaton(G):
    assert len(G.startSymbol.productions) == 1, 'Grammar must be augmented'

    start_production = G.startSymbol.productions[0]
    start_item = Item(start_production, 0)

    automaton = State(start_item, True)

    pending = [ start_item ]
    visited = { start_item: automaton }

    while pending:
        current_item = pending.pop()
        if current_item.IsReduceItem:
            continue

        next_symbol = current_item.NextSymbol
        next_item = current_item.NextItem()

        if not next_item in visited:
            pending.append(next_item)
            visited[next_item] = State(next_item, True)

        # Your code here!!! (Decide which transitions to add)
        if next_symbol.IsNonTerminal:
            for prod in next_symbol.productions:
                other_item = Item(prod, 0)
                if not other_item in visited:
                    pending.append(other_item)
                    visited[other_item] = State(other_item, True)

        current_state = visited[current_item]
        # Your code here!!! (Add the decided transitions)

        current_state.add_transition(next_symbol.name, visited[next_item])

        if next_symbol.IsNonTerminal:
            for prod in next_symbol.productions:
                current_state.add_epsilon_transition(visited[Item(prod, 0)])

    return automaton


class ShiftReduceParser:
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'
    OK = 'OK'

    def __init__(self, G, verbose=False):
        self.G = G
        self.verbose = verbose
        self.action = {}
        self.goto = {}
        self._build_parsing_table()

    def _build_parsing_table(self):
        raise NotImplementedError()

    def __call__(self, w) -> Tuple[List[grammars_utils.Production], grammars_utils.DerivationTree]:
        stack = [(0, None)]
        cursor = 0
        output = []
        table = self.action
        self.action = modify_tablelr1(self.action)
        print(w)
        while True:
            state = stack[-1][0]
            lookahead = w[cursor]
            if self.verbose: print(stack, w[cursor:])

            # Your code here!!! (Detect error)
            try:
                action, tag = self.action[state, lookahead][0]
                # Your code here!!! (Shift case)
                if action == ShiftReduceParser.SHIFT:
                    stack.append((tag, grammars_utils.DerivationTree(
                        grammars_utils.Symbol(lookahead, None))))
                    cursor += 1
                # Your code here!!! (Reduce case)
                elif action == ShiftReduceParser.REDUCE:
                    sons = []
                    for _ in range(len(tag.Right)):
                        sons.append(stack.pop()[1])

                    tree = grammars_utils.DerivationTree(tag.Left)
                    tree.sons = sons

                    stack.append((self.goto[stack[-1][0], tag.Left][0], tree))
                    output.append(tag)
                # Your code here!!! (OK case)
                elif action == ShiftReduceParser.OK:
                    return output, stack[-1][1]
                # Your code here!!! (Invalid case)
                else:
                    assert False, 'Must be something wrong!'
            except KeyError:
                raise Exception('Aborting parsing, item is not viable.')


def expand(item, firsts):
    next_symbol = item.NextSymbol
    if next_symbol is None or not next_symbol.IsNonTerminal:
        return []

    lookaheads = ContainerSet()
    # Your code here!!! (Compute lookahead for child items)
    for preview in item.Preview():
        lookaheads.hard_update(compute_local_first(firsts, preview))

    assert not lookaheads.contains_epsilon
    # Your code here!!! (Build and return child items)
    return [Item(prod, 0, lookaheads) for prod in next_symbol.productions]


def compress(items):
    centers = {}

    for item in items:
        center = item.Center()
        try:
            lookaheads = centers[center]
        except KeyError:
            centers[center] = lookaheads = set()
        lookaheads.update(item.lookaheads)

    return {Item(x.production, x.pos, set(lookahead)) for x, lookahead in
            centers.items()}


def closure_lr1(items, firsts):
    closure = ContainerSet(*items)

    changed = True
    while changed:
        changed = False

        new_items = ContainerSet()
        # Your code here!!!
        for item in closure:
            new_items.extend(expand(item, firsts))

        changed = closure.update(new_items)

    return compress(closure)


def goto_lr1(items, symbol, firsts=None, just_kernel=False):
    assert just_kernel or firsts is not None, '`firsts` must be provided if `just_kernel=False`'
    items = frozenset(item.NextItem() for item in items if item.NextSymbol == symbol)
    return items if just_kernel else closure_lr1(items, firsts)


def build_LR1_automaton(G):
    assert len(G.startSymbol.productions) == 1, 'Grammar must be augmented'

    firsts = compute_firsts(G)
    firsts[G.EOF] = ContainerSet(G.EOF)

    start_production = G.startSymbol.productions[0]
    start_item = Item(start_production, 0, lookaheads=(G.EOF,))
    start = frozenset([start_item])

    closure = closure_lr1(start, firsts)
    automaton = State(frozenset(closure), True)

    pending = [start]
    visited = {start: automaton}

    while pending:
        current = pending.pop()
        current_state = visited[current]

        for symbol in G.terminals + G.nonTerminals:
            # Your code here!!! (Get/Build `next_state`)
            kernels = goto_lr1(current_state.state, symbol, just_kernel=True)

            if not kernels:
                continue

            try:
                next_state = visited[kernels]
            except KeyError:
                pending.append(kernels)
                visited[pending[-1]] = next_state = State(
                    frozenset(goto_lr1(current_state.state, symbol, firsts)),
                    True)

            current_state.add_transition(symbol.Name, next_state)

    automaton.set_formatter(multiline_formatter)
    return automaton



class LR1Parser(ShiftReduceParser):

    def _build_parsing_table(self):
        G = self.G.AugmentedGrammar(True)

        automaton = build_LR1_automaton(G)
        for i, node in enumerate(automaton):
            if self.verbose: print(i, '\t',
                                   '\n\t '.join(str(x) for x in node.state),
                                   '\n')
            node.idx = i

        node: automaton_utils.State
        for node in automaton:
            idx = node.idx
            for item in node.state:
                # Your code here!!!
                # - Fill `self.Action` and `self.Goto` according to `item`)
                # - Feel free to use `self._register(...)`)
                if item.IsReduceItem:
                    prod = item.production
                    if prod.Left == G.startSymbol:
                        LR1Parser._register(self.action, (idx, G.EOF),
                                            (ShiftReduceParser.OK, None))
                    else:
                        for lookahead in item.lookaheads:
                            LR1Parser._register(self.action, (idx, lookahead), (
                            ShiftReduceParser.REDUCE, prod))
                else:
                    next_symbol = item.NextSymbol
                    if next_symbol.IsTerminal:
                        LR1Parser._register(self.action, (idx, next_symbol), (
                        ShiftReduceParser.SHIFT, node[next_symbol.Name][0].idx))
                    else:
                        LR1Parser._register(self.goto, (idx, next_symbol),
                                            node[next_symbol.Name][0].idx)
                pass

    @staticmethod
    def _register(table, key, value):
        # assert key not in table or table[
        #     key] == value, 'Shift-Reduce or Reduce-Reduce conflict!!!'
        # table[key] = value
        if not table.get(key, False):
            table[key] = [value]
        else: table[key].append(value)


def modify_tablelr1(table):
    new_table = {}
    for (noterminal, symbol), action in table.items():
        new_table[noterminal, str(symbol)] = action

    return new_table



if __name__ == '__main__':
   #  G = grammars_utils.Grammar()
   #  E = G.NonTerminal('E', True)
   #  A = G.NonTerminal('A')
   #  equal, plus, num = G.Terminals('= + int')
   #
   #  E %= A + equal + A | num
   #  A %= num + plus + A | num
   #
   #  parser = LR1Parser(G, True)
   #  derivation = parser([''])
   #
   #  tree = derivation[1]
   #  tree.graph().write_svg(path=r'.\some', prog=r'C:\Program Files (x86)\Graphviz2.38\bin\dot.exe')
   #

    gra = grammars_utils.build_grammar(['E->SS',
'S->CC',
'C->cC',
'C->d'])
    parser = LR1Parser(gra)

    print(parser([sim for sim in 'cccdcdccdcd']+['$']))









