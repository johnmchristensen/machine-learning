import random

from sympy import false


class Grammar:
    def __init__(self):
        self._start = Edge()
        self._nodes = [self.start]

    def create_node(self):
        node = Node()
        self._nodes.append(node)
        return node

    def add_edge(self, from_node, letter, to_node):
        if from_node not in self._nodes:
            raise GrammarException("from_node not in grammar")

        if to_node not in self._nodes:
            raise GrammarException("to_node not in grammar")

        from_node.add_edge(letter, to_node)

    def produce(self):
        current = self._start
        while current is not None:
            yield current.Letter
            current = current.Target.navigate()

    def is_valid(self, string):
        if len(string) == 0:
            return True

        if self._start.Letter != string[0]:
            return False

        current = self._start.Target
        for letter in string[1:]:
            success, current = current.try_navigate(letter)
            if not success:
                return false

        return True


class Edge:
    def _init(self, letter, target):
        self.Letter = letter
        self.Target = target

class Node:
    def __init__(self):
        self._edges = []

    def add_edge(self, edge):
        self._edges.append(edge)

    def navigate(self):
        if len(self._edges) == 0:
            return None

        if len(self._edges) == 1:
            return self._edges[0]

        return self._edges[random.randint(0,len(self._edges)-1)]

    def try_navigate(self, letter):
        edge = next((e for e in self._edges if e.Letter == letter), None)
        if edge is None:
            return False, None

        return True, edge.Target


class GrammarException(Exception):
    pass