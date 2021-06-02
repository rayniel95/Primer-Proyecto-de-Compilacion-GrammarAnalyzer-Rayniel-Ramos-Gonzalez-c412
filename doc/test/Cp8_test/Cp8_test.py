from cmp.tools.automata import NFA, DFA, nfa_to_dfa
from cmp.tools.automata import automata_union, automata_concatenation
from cmp.tools.automata import automata_closure, automata_minimization
from cmp.ast import get_printer
from cmp.utils import Token
from cmp.pycompiler import Grammar


EPSILON = 'ε'


class Node:
    def evaluate(self) -> NFA:
        raise NotImplementedError()

# todo lex is lexeme?????????
class AtomicNode(Node):
    def __init__(self, lex: str):
        self.lex = lex


class UnaryNode(Node):
    def __init__(self, node: 'Node'):
        self.node = node

    def evaluate(self):
        value = self.node.evaluate()
        return self.operate(value)

    @staticmethod
    def operate(value: NFA) -> NFA:
        raise NotImplementedError()


class BinaryNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self):
        lvalue = self.left.evaluate()
        rvalue = self.right.evaluate()
        return self.operate(lvalue, rvalue)

    @staticmethod
    def operate(lvalue, rvalue):
        raise NotImplementedError()


class EpsilonNode(AtomicNode):
    def evaluate(self):
        # Your code here!!!
        return DFA(states=1, finals=[0], transitions={})

# EpsilonNode(EPSILON).evaluate()
# printer = get_printer(AtomicNode=AtomicNode, UnaryNode=UnaryNode,
#                       BinaryNode=BinaryNode)
#
class SymbolNode(AtomicNode):
    def evaluate(self):
        s = self.lex
        # Your code here!!!
        return DFA(states=2, finals=[1], transitions={(0, s): 1})

# SymbolNode('a').evaluate()

class ClosureNode(UnaryNode):
    @staticmethod
    def operate(value: NFA):
        # Your code here!!!
        return automata_closure(value)


# ClosureNode(SymbolNode('a')).evaluate()


class UnionNode(BinaryNode):
    @staticmethod
    def operate(lvalue: NFA, rvalue: NFA):
        # Your code here!!!
        return automata_union(lvalue, rvalue)


# UnionNode(SymbolNode('a'), SymbolNode('b')).evaluate()


class ConcatNode(BinaryNode):
    @staticmethod
    def operate(lvalue, rvalue):
        # Your code here!!!
        return automata_concatenation(lvalue, rvalue)

# ConcatNode(SymbolNode('a'), SymbolNode('b')).evaluate()


G = Grammar()

E = G.NonTerminal('E', True)
T, F, A, X, Y, Z = G.NonTerminals('T F A X Y Z')
pipe, star, opar, cpar, symbol, epsilon = G.Terminals('| * ( ) symbol ε')

# # > PRODUCTIONS???
# # Your code here!!!
# # el | es el de menor prioridad, concat, clousure, s
#
# print(G)
#

#
#
# def regex_tokenizer(text, G, skip_whitespaces=True):
#     tokens = []
#     # > fixed_tokens = ???
#     # Your code here!!!
#     for char in text:
#         if skip_whitespaces and char.isspace():
#             continue
#         # Your code here!!!
#
#     tokens.append(Token('$', G.EOF))
#     return tokens
#
#
# tokens = regex_tokenizer('a*(a|b)*cd | ε', G)
# tokens
#
#
# from cmp.tools.parsing import metodo_predictivo_no_recursivo
#
# parser = metodo_predictivo_no_recursivo(G)
# left_parse = parser(tokens)
# left_parse