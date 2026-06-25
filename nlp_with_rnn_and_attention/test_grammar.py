import os
import sys
import unittest
from unittest.mock import patch

# Allow importing sibling module `grammar.py` when run from workspace root.
sys.path.insert(0, os.path.dirname(__file__))

from grammar import Edge, Grammar, GrammarException, Node


class TestNodeNavigate(unittest.TestCase):
    def test_navigate_returns_none_when_no_edges(self):
        node = Node()
        self.assertIsNone(node.navigate())

    def test_navigate_returns_only_edge_when_single_edge(self):
        node = Node()
        edge = object()
        node.add_edge(edge)

        self.assertIs(node.navigate(), edge)

    def test_navigate_uses_random_index_when_multiple_edges(self):
        node = Node()
        edges = ["first", "second", "third"]
        for edge in edges:
            node.add_edge(edge)

        with patch("grammar.random.randint", return_value=1) as mocked_randint:
            chosen = node.navigate()

        mocked_randint.assert_called_once_with(0, 2)
        self.assertEqual(chosen, "second")


class TestGrammarAddEdgeValidation(unittest.TestCase):
    def _grammar_without_constructor(self):
        grammar = Grammar.__new__(Grammar)
        grammar._nodes = []
        return grammar

    def test_add_edge_raises_when_from_node_not_in_grammar(self):
        grammar = self._grammar_without_constructor()
        to_node = Node()
        grammar._nodes = [to_node]

        with self.assertRaisesRegex(GrammarException, "from_node not in grammar"):
            grammar.add_edge(Node(), "A", to_node)

    def test_add_edge_raises_when_to_node_not_in_grammar(self):
        grammar = self._grammar_without_constructor()
        from_node = Node()
        grammar._nodes = [from_node]

        with self.assertRaisesRegex(GrammarException, "to_node not in grammar"):
            grammar.add_edge(from_node, "A", Node())


class TestGrammarIsValid(unittest.TestCase):
    def _make_edge(self, letter, target):
        # `Edge` currently has `_init` instead of `__init__`, so tests set fields directly.
        edge = Edge()
        edge.Letter = letter
        edge.Target = target
        return edge

    def _make_grammar(self, start_letter="A"):
        grammar = Grammar.__new__(Grammar)
        first_node = Node()
        grammar._start = self._make_edge(start_letter, first_node)
        grammar._nodes = [first_node]
        return grammar, first_node

    def test_is_valid_returns_false_on_first_letter_mismatch(self):
        grammar, _ = self._make_grammar(start_letter="A")

        self.assertFalse(grammar.is_valid("B"))

    def test_is_valid_returns_true_for_matching_single_character(self):
        grammar, _ = self._make_grammar(start_letter="A")

        self.assertTrue(grammar.is_valid("A"))

    def test_is_valid_returns_true_for_two_character_valid_path(self):
        grammar, first_node = self._make_grammar(start_letter="A")
        first_node.add_edge(self._make_edge("B", Node()))

        self.assertTrue(grammar.is_valid("AB"))


class TestGrammarIsValidKnownDefects(unittest.TestCase):
    def _make_edge(self, letter, target):
        edge = Edge()
        edge.Letter = letter
        edge.Target = target
        return edge

    def _make_grammar(self, start_letter="A"):
        grammar = Grammar.__new__(Grammar)
        first_node = Node()
        grammar._start = self._make_edge(start_letter, first_node)
        grammar._nodes = [first_node]
        return grammar, first_node

    def test_is_valid_returns_false_when_transition_does_not_exist(self):
        grammar, _ = self._make_grammar(start_letter="A")

        self.assertFalse(grammar.is_valid("AB"))

    def test_is_valid_handles_empty_string(self):
        grammar, _ = self._make_grammar(start_letter="A")

        self.assertTrue(grammar.is_valid(""))

    def test_is_valid_supports_multi_step_path(self):
        grammar, first_node = self._make_grammar(start_letter="A")
        second_node = Node()
        terminal_node = Node()
        first_node.add_edge(self._make_edge("B", second_node))
        second_node.add_edge(self._make_edge("C", terminal_node))

        self.assertTrue(grammar.is_valid("ABC"))

if __name__ == "__main__":
    unittest.main()

