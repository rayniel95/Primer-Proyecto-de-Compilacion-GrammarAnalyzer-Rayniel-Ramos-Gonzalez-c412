from typing import Dict, Tuple, List
from flask import Flask, request, render_template
from ply import yacc

import grammars_utils
import first_follow
import automaton_utils
import lr1_parser
import slr_lalr_parser


app = Flask(__name__)
graphviz_path = None


@app.route('/', methods=['GET', 'POST'])
def analyzer():
    global graphviz_path
    if request.method == 'GET' and not graphviz_path:
        # devolver el formulario para introducir el path
        return render_template('index.html')
    if request.method == 'GET':
        # devolver el formulario para introducir la gramarica
        return render_template('ga_ui.html', first={}, follows={})
    if request.method == 'POST' and not graphviz_path:
        # setear el path y devolver el formulario para introducir la gramatica
        graphviz_path = request.form['graphviz_path']
        return render_template('ga_ui.html', first={}, follows={},
                               factorized={'terminals': [],
                                           'noterminals': [],
                                           'productions': []},
                               is_regular=False, re='', fnc=[],
                               trees=[], is_ll1=False,
                               rows=[], cols=[], table=[])

    # devolver el formulario con la respuesta de la gramatica
    if not request.form.get('grammar', False):
        return 'Debes introducir al menos la gramatica'

    grammar = request.form['grammar'].replace('\r', '').replace(' ', '').split('\n')
    print(grammar)
    gra = grammars_utils.build_grammar(grammar)
    firsts = first_follow.compute_firsts(gra)
    follows = first_follow.compute_follows(gra, firsts)
    terminals_dlr, noterminal_dlr, productions_dlr = drop_left_recursivity(gra)

    fact = factorize_grammar(gra)
    regular, re = regular_grammar(gra)
    chomsky = fnc(gra)
    table = None
    trees = []
    ll1 = False
    action_rows = None
    action_cols = None
    action_table = None
    goto_table = {}
    goto_cols = {}
    goto_rows = {}
    sr_conflicts = {}
    rr_conflicts = {}


    if request.form.get('try_ll1', False):
        # aplicar ll1, mostrar la tabla y los arboles
        ll1, table = is_ll1(gra, firsts, follows)

        if ll1 and request.form.get('words', False):
            words = request.form['words'].replace('\r', '').split('\n')

            tree_list = ll1_der_trees(words, table, firsts,
                                 follows, gra)
            trees = do_svg(tree_list)
        action_table, action_rows, action_cols = table2str(table)


    if request.form.get('try_slr1', False):
        g, my_lex = slr_lalr_parser.build_grammar(grammar)

        table = yacc.LRGeneratedTable(g, 'SLR')
        parser = yacc.LRParser(table, slr_lalr_parser.err_func)

        if request.form.get('words', False):
            words = request.form['words'].replace('\r', '').split('\n')

            tree_list = slr_lalr_der_trees(words, parser, my_lex)
            trees = do_svg(tree_list)

        action_table, action_rows, action_cols = action_table2str_lalr(
            table.lr_action, g)

        goto_table, goto_rows, goto_cols = goto_table2str_slr(
            table.lr_goto)
        sr_conflicts = table.sr_conflicts
        rr_conflicts = table.rr_conflicts

    if request.form.get('try_lr1', False):
        parser = lr1_parser.LR1Parser(gra)
        lr1, action, goto = is_lr1(gra, parser)

        if lr1 and request.form.get('words', False):
            words = request.form['words'].replace('\r', '').split('\n')
            print(words)
            tree_list = lr1_der_trees(words, parser)
            trees = do_svg(tree_list)
            print(trees)

        action_table, action_rows, action_cols = table2str(action)
        goto_table, goto_rows, goto_cols = table2str(goto)

    if request.form.get('try_lalr1', False):
        g, my_lex = slr_lalr_parser.build_grammar(grammar)

        table = yacc.LRGeneratedTable(g)
        parser = yacc.LRParser(table, slr_lalr_parser.err_func)

        if request.form.get('words', False):
            words = request.form['words'].replace('\r', '').split('\n')
            tree_list = slr_lalr_der_trees(words, parser, my_lex)
            trees = do_svg(tree_list)

        action_table, action_rows, action_cols = action_table2str_lalr(
            table.lr_action, g)

        goto_table, goto_rows, goto_cols = goto_table2str_slr(
            table.lr_goto)

        sr_conflicts = table.sr_conflicts
        rr_conflicts = table.rr_conflicts

    return render_template('ga_ui.html', first=firsts, follows=follows,
        factorized={'terminals': fact[0], 'noterminals': fact[1],
                    'productions': fact[2]},
        is_regular=regular, re=re, fnc=chomsky, trees=trees, is_ll1=ll1,
        action_rows=action_rows, action_cols=action_cols,
        action_table=action_table,
        goto_table=goto_table, goto_rows=goto_rows, goto_cols=goto_cols,
        terminals_dlr=terminals_dlr, noterminals_dlr=noterminal_dlr,
        productions_dlr=productions_dlr, sr_conflicts=sr_conflicts,
        rr_conflicts=rr_conflicts)


# def compute_firsts(grammar) -> Dict[str, str]:
#     firsts = first_follow.compute_firsts(grammar)
#     fi = {}
#     for symb, first in firsts.items():
#         if isinstance(symb, grammars_utils.Terminal) or \
#             isinstance(symb, grammars_utils.NonTerminal):
#             fi[symb.Name] = ' '.join(str(symbol) for symbol in first)
#     return fi


# def compute_follows(grammar, firsts) -> Dict[str, str]:
#     follows = first_follow.compute_follows(grammar, firsts)
#     print(follows)
#     fo = {}
#     for symb, follow in follows.items():
#         fo[symb.Name] = ' '.join(str(symbol) for symbol in follow)
#     return fo


def factorize_grammar(grammar) -> Tuple[str, str, List[str]]:
    g = grammars_utils.factorize_grammar(grammar)

    terminals = ' '.join(str(term) for term in g.terminals)
    noterminals = ' '.join(str(noter) for noter in g.nonTerminals)
    productions = [str(pro).replace(':=', '->') for pro in g.Productions]
    return (terminals, noterminals, productions)


def regular_grammar(grammar) -> Tuple[bool, str]:
    if grammars_utils.is_regular_grammar(grammar):
        dfa = automaton_utils.reg_grammar2DFA(grammar)
        dfa.graph().write_svg(path=r'.\static\automaton.svg', prog=graphviz_path)
        return True, automaton_utils.automaton2reg(dfa)

    return False, ''


def fnc(grammar) -> List[str]:
    return [str(prod).replace(':=', '->')
            for prod in grammars_utils.glc2fnc(grammar).Productions]


def is_ll1(grammar, first, follow) -> Tuple[bool, dict]:
    table = first_follow.build_parsing_table(grammar, first, follow)

    for row_column, value in table.items():
        if len(value) > 1: return False, table

    return True, table


def ll1_der_trees(words: List[str], table: dict, first, follow, grammar):
    new_table = grammars_utils.modify_tablell1(table)
    parser = first_follow.metodo_predictivo_no_recursivo(grammar, new_table, first,
                                                         follow)

    deriv_trees = []
    for word in words:
        tree = parser([simbol for simbol in word])
        if tree:
            deriv_trees.append(tree[1])

    return deriv_trees


def is_lr1(grammar, parser) -> Tuple[bool, dict, dict]:
    for item, value in parser.action:
        if len(value) > 1: return False, parser.action, parser.goto

    else: return True, parser.action, parser.goto


def lr1_der_trees(words: List[str], parser: lr1_parser.LR1Parser) -> List[grammars_utils.DerivationTree]:
    trees = []
    for word in words:
        derivation, tree = parser([symbo for symbo in word] + ['$'])
        trees.append(tree)

    return trees


def slr_lalr_der_trees(words: List[str], parser, lexer):
    trees = []
    for word in words:
        trees.append(parser.parse(word, lexer))

    return trees


def do_svg(trees):
    trees_name = []
    count = 1
    for tree in trees:
        name = f'tree{count}.svg'
        tree_name = f'./static/{name}'
        tree.graph().write_svg(path=tree_name, prog=graphviz_path)
        trees_name.append(name)
        count += 1
    return trees_name


def goto_table2str_slr(goto_table) -> Tuple[Dict[Tuple[str, str], str], List[str], List[str]]:
    rows = set()
    cols = set()
    new_table = {}
    for state in goto_table:
        rows.add(str(state))
        for symbol in goto_table[state]:
            cols.add(str(symbol))
            new_table[str(state), str(symbol)] = str(goto_table[state][symbol])

    for row in rows:
        for col in cols:
            if not new_table.get((row, col,), False):
                new_table[row, col] = 'NaN'

    return new_table, sorted(list(rows), key=lambda x: int(x)), \
           sorted(list(cols))


def table2str(table):
    rows = set()
    cols = set()
    new_table = {}
    for row, col in table:
        rows.add(str(row))
        cols.add(str(col))
        val = table[row, col]
        new_table[str(row), str(col)] = str(val).replace(':=', '->')

    for row in rows:
        for col in cols:
            if not new_table.get((row, col,), False):
                new_table[row, col] = 'NaN'

    return new_table, sorted(list(rows)), sorted(list(cols))


def action_table2str_lalr(action_table, grammar: yacc.Grammar)-> Tuple[Dict[Tuple[str, str], str], List[str], List[str]]:
    print(action_table)
    rows = set()
    cols = set()
    new_table = {}
    for state in action_table:
        rows.add(str(state))
        for symbol in action_table[state]:
            cols.add(str(symbol))
            value = action_table[state][symbol]
            if  value < 0:
                new_table[str(state), str(symbol)] = str(grammar[-value])
            else:
                new_table[str(state), str(symbol)] = str(action_table[state][symbol])

    for row in rows:
        for col in cols:
            if not new_table.get((row, col,), False):
                new_table[row, col] = 'NaN'

    return new_table, sorted(list(rows), key=lambda x: int(x)), \
           sorted(list(cols))


def drop_left_recursivity(grammar):
    g = grammars_utils.delete_left_recursivity(grammar)
    terminals = ' '.join(str(term) for term in g.terminals)
    noterminals = ' '.join(str(noter) for noter in g.nonTerminals)
    productions = [str(pro).replace(':=', '->') for pro in g.Productions]
    return (terminals, noterminals, productions)




app.run(debug=True)


# todo ademas de que se vean el epsilon en las derivaciones intermedias del
#  arbol para algunas derivaciones no se ve

# todo usar set en vez de listas en los parsers ll1 lr1

# todo se esta ordenando con str a la hora de pasar de tabla para la
#  representacion a str