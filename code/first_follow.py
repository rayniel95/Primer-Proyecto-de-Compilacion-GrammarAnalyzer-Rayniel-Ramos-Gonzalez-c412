from itertools import islice
from pprint import pprint
from typing import Dict, Union, Tuple
import grammars_utils
from grammars_utils import ContainerSet, NonTerminal, Grammar, \
    Sentence, Symbol, Production, DerivationTree, build_grammar, modify_tablell1


# Computes First(alpha), given First(Vt) and First(Vn)
# alpha in (Vt U Vn)*
def compute_local_first(firsts: Dict[Union[grammars_utils.Terminal, NonTerminal], ContainerSet],
                        alpha: Sentence) -> ContainerSet:
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
    # epsilon pertenece a First(Xi) ? First(Xi+1) subconjunto de
    # First(X) y First(alpha)
    # epsilon pertenece a First(X1)...First(XN) ? epsilon pertence a First(X) y
    # al First(alpha)
    ###################################################
    #                   <CODE_HERE>                   #
    if isinstance(alpha[0], grammars_utils.Terminal):
        first_alpha.add(alpha[0])
        return first_alpha

    for symbol in alpha:
        first_alpha.update(firsts[symbol])
        #  todo y qe pasa cuando hay terminales en el medio de la forma oracional???
        #   necesito esto para que la tabla este bien
        if not firsts[symbol].contains_epsilon:
            return first_alpha
    ###################################################
    # first(alpha) contiene e solo si Xn contiene e
    first_alpha.set_epsilon()
    # First(alpha)
    return first_alpha

# Computes First(Vt) U First(Vn) U First(alpha)
# P: X -> alpha
def compute_firsts(G) -> Dict[Symbol, ContainerSet]:
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


def compute_follows(G: 'Grammar', firsts: dict) -> Dict[Symbol, ContainerSet]:
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
            alpha: Sentence = production.Right

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
                        #rest = islice(alpha, index + 1, len(alpha))
                        rest = alpha[index + 1:]
                        if firsts.get(Sentence(*rest), False):
                            first_rest = firsts[Sentence(*rest)]
                        elif local_firsts.get(Sentence(*rest), False):
                            first_rest = local_firsts[Sentence(*rest)]

                        else:
                            first_rest = compute_local_first(firsts, rest)
                            local_firsts[Sentence(*rest)] = first_rest

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
                if not M.get((X, terminal), False):
                    M[X, terminal] = [production]
                else: M[X, terminal].append(production)

            if firsts[alpha].contains_epsilon and terminal in follows[X]:
                if not M.get((X, terminal), False):
                    M[X, terminal] = [production]
                else: M[X, terminal].append(production)

        # todo esto no tiene sentido
        if G.EOF in firsts[alpha]:
            if M.get((X, G.EOF), False):
                return {}

            M[X, G.EOF] = [production]

        if firsts[alpha].contains_epsilon and G.EOF in follows[X]:
            if not M.get((X, G.EOF), False):
                M[X, G.EOF] = [production]
            else: M[X, G.EOF].append(production)
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
        tree = DerivationTree(G.startSymbol)
        stack = [(G.EOF, None,), (G.startSymbol, tree,)]
        cursor = 0
        output = []  # las producciones aplicadas
        ###################################################

        # parsing w...
        while True:
            top, node = stack.pop()
            a = w[cursor]
        #             print((top, a))

        ###################################################
        #                   <CODE_HERE>                   #
        ###################################################
            if isinstance(top, (grammars_utils.Terminal,)):
                if a != str(top): return []
            # todo que pasa con e?????? si es e se popea y se sigue
                cursor += 1
                if cursor >= len(w): break

            else:
                if not M.get((top, a), False): return []
                output.append(M[top, a][0])

                sentence: Sentence = M[top, a][0].Right
                for symbol in reversed(sentence):
                    son = DerivationTree(symbol, node)

                    stack.append((symbol, son,))
                    node.add_son(son)

                if isinstance(sentence, grammars_utils.Epsilon):
                    node.add_son(DerivationTree(grammars_utils.Epsilon(None),
                                                node))
        # left parse is ready!!!
        return (output, tree,)

    # parser is ready!!!
    return parser


# algo = Symbol('name', Grammar())
#
# while algo.IsEpsilon:
#     continue


# region test1
#
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
#
# firsts = compute_firsts(G)
#
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
#
# follows = compute_follows(G, firsts)
#
# print(follows)
#
# assert follows == {
#    E: ContainerSet(G.EOF, cpar , contains_epsilon=False),
#    T: ContainerSet(cpar, plus, G.EOF, minus , contains_epsilon=False),
#    F: ContainerSet(cpar, star, G.EOF, minus, div, plus , contains_epsilon=False),
#    X: ContainerSet(G.EOF, cpar , contains_epsilon=False),
#    Y: ContainerSet(cpar, plus, G.EOF, minus , contains_epsilon=False)
# }
#
# M = build_parsing_table(G, firsts, follows)
#
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
#
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

if __name__ == '__main__':
    g1 = ['S->A=A', 'A->a+A', 'A->a']
    g2 = ['S->aS', 'S->bS', 'S->', 'A->aS']
    g3 = ['S->AB', 'A->aT', 'T->A', 'T->', 'B->bK', 'K->B', 'K->']
    #
    g = build_grammar(g3)
    first = compute_firsts(g)
    follow = compute_follows(g, first)
    print(follow)
    # table = build_parsing_table(g, first, follow)
    # new_table = modify_tablell1(table)
    # parser = metodo_predictivo_no_recursivo(g, new_table, first,
    #                                                      follow)
    # print(table)
    #
    # print(parser([si for si in 'ababaaaaaa']))