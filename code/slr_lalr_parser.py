from typing import List, Union, Tuple
import ply.yacc as yacc
import ply.lex as lex
from grammars_utils import DerivationTree, Symbol


def build_grammar(productions: List[str]) -> Tuple[yacc.Grammar, lex.Lexer]:
    '''
    :param productions: lista de producciones sin espacio
    :type productions:
    :return:
    :rtype:
    '''
    terminals = []
    for prod in productions:
        parts = prod.split('->')
        for symbol in parts[1]:
            if not symbol.isupper(): terminals.append(symbol)

    grammar = yacc.Grammar(terminals)

    for prod in productions:
        parts = prod.split('->')
        symbols = []
        for symbol in parts[1]:
            if symbol in terminals:
                symbols.append(f'"{symbol}"')
            else: symbols.append(symbol)
        grammar.add_production(parts[0], symbols, func='der_tree')

    grammar.set_start()

    prod: yacc.Production
    for prod in grammar.Productions:
        prod.bind({'der_tree': der_tree})

    grammar.build_lritems()

    my_lex = lex.Lexer()
    my_lex.lexliterals = terminals
    my_lex.lexre = []

    return grammar, my_lex


def der_tree(p):
    p: yacc.YaccProduction
    p.slice: List[Union[yacc.YaccSymbol, lex.LexToken]]

    tree = DerivationTree(Symbol(p.slice[0].type, None))

    if len(p) == 1:
        tree.add_son(DerivationTree(Symbol('epsilon', None)))
    else:
        for index in range(len(p[1:])):
            if type(p[index + 1]) == str:
                tree.add_son(DerivationTree(Symbol(p[index + 1], None)))
            else:
                tree.add_son(p[index + 1])
    tree.sons.reverse()
    p[0] = tree



def build_lr_table(grammar: yacc.Grammar, method: str):
    assert method == 'SLR' or method == 'LALR'

    table = yacc.LRGeneratedTable(grammar, method)
    # todo ya aqui estan todas las tablas y conflictos

def err_func(parse):
    """
    Error rule for Syntax Errors handling and reporting.
    """
    if parse is None:
        print("Error! Unexpected end of input!")
    else:
        print("Syntax error! Line: {}, position: {}, character: {}, "
              "type: {}".format(parse.lineno, parse.lexpos, parse.value,
                                parse.type))

        print(parse)


if __name__ == '__main__':
    g, my_lex = build_grammar(['E->TX', 'X->+TX', 'X->P', 'T->FY', 'Y->*FY', 'T->P',
                       'F->(E)', 'F->i', 'P->', 'Y->P'])

    table = yacc.LRGeneratedTable(g, 'SLR')
    parser = yacc.LRParser(table, err_func)
    print(table.lr_action)
    print()
    print(table.lr_goto)
    tree: DerivationTree
    tree = parser.parse('i+i*i', my_lex)
    tree.print_tree()
    tree.graph().write_svg(path=r'./some', prog=r'C:\Program Files (x86)\Graphviz2.38\bin\dot.exe')
    # todo falta mostrar la tabla

    print(table.rr_conflicts)
    print(table.sr_conflicts)