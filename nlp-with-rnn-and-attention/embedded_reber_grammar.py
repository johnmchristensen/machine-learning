import random

from sparse_grammar import SparseGrammar

class EmbeddedReberGrammar:
    def __init__(self):
        self._reber = ReberGrammar()

    def produce(self):
        yield 'B'
        is_top = random.randint(0, 1) == 0
        yield 'T' if is_top else 'P'
        yield from self._reber.produce()
        yield 'T' if is_top else 'P'
        yield 'E'

class ReberGrammar(SparseGrammar):
    def __init__(self):
        super().__init__(6, 'B', 'E')
        self.add_edge(0, 1, 'T')
        self.add_edge(0, 4, 'P')
        self.add_edge(1, 1, 'S')
        self.add_edge(1, 2, 'X')
        self.add_edge(2, 3, 'S')
        self.add_edge(2, 4, 'X')
        self.add_edge(4, 4, 'T')
        self.add_edge(4, 5, 'V')
        self.add_edge(5, 2, 'P')
        self.add_edge(5, 3, 'V')