import os
import sys
import unittest
from unittest.mock import patch

# Allow importing sibling module `sparse_grammar.py` when run from workspace root.
sys.path.insert(0, os.path.dirname(__file__))

from sparse_grammar import SparseGrammar


class TestSparseGrammarAddEdge(unittest.TestCase):
    def test_add_edge_sets_symbol_at_source_destination(self):
        grammar = SparseGrammar(2, 'Z', 'X')

        grammar.add_edge(0, 1, "A")

        self.assertEqual(grammar._edges[0][1], "A")

    def test_add_edge_overwrites_existing_symbol(self):
        grammar = SparseGrammar(2, 'Z', 'X')

        grammar.add_edge(0, 1, 'A')
        grammar.add_edge(0, 1, "B")

        self.assertEqual(grammar._edges[0][1], "B")

    def test_add_edge_raises_index_error_when_source_is_out_of_bounds(self):
        grammar = SparseGrammar(2, 'Z', 'X')

        with self.assertRaises(IndexError):
            grammar.add_edge(9, 0, "A")

    def test_produce_yields_path_symbols_until_terminal(self):
        grammar = SparseGrammar(3, 'Z' , 'X')
        grammar.add_edge(0, 1, "A")
        grammar.add_edge(1, 2, "B")

        with patch("sparse_grammar.random.randint", return_value=0):
            output = list(grammar.produce())

        self.assertEqual(output, ["Z", "A", "B", "X"])


