import os
import sys
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

# Allow importing sibling module `reber_dataset.py` when run from workspace root.
sys.path.insert(0, os.path.dirname(__file__))

from reber_dataset import ReberDataset


class TestReberDataset(unittest.TestCase):
    @patch("reber_dataset.random.choice", return_value="B")
    @patch("reber_dataset.random.randint", return_value=1)
    @patch("reber_dataset.EmbeddedReberGrammar")
    def test_len_and_labels_for_even_length(self, embedded_cls, _randint, _choice):
        embedded_cls.return_value.produce.return_value = ["B", "T", "E"]

        dataset = ReberDataset(6)

        self.assertEqual(len(dataset), 6)

        valid_tokens, valid_label = dataset[0]
        expected_valid = torch.tensor([dataset.TOKEN_TO_ID[token] for token in ["B", "T", "E"]], dtype=torch.long)
        self.assertTrue(torch.equal(valid_tokens, expected_valid))
        self.assertTrue(torch.equal(valid_label, torch.tensor([1.0], dtype=torch.float32)))

        invalid_tokens, invalid_label = dataset[3]
        expected_invalid = torch.tensor([dataset.TOKEN_TO_ID[token] for token in ["B", "P", "E"]], dtype=torch.long)
        self.assertTrue(torch.equal(invalid_tokens, expected_invalid))
        self.assertTrue(torch.equal(invalid_label, torch.tensor([0.0], dtype=torch.float32)))

    @patch("reber_dataset.random.choice", return_value="B")
    @patch("reber_dataset.random.randint", return_value=1)
    @patch("reber_dataset.EmbeddedReberGrammar")
    def test_len_for_odd_length_uses_integer_half(self, embedded_cls, _randint, _choice):
        embedded_cls.return_value.produce.return_value = ["B", "P", "E"]

        dataset = ReberDataset(5)

        # Current implementation builds int(length / 2) valid and invalid items.
        self.assertEqual(len(dataset), 4)

    @patch("reber_dataset.random.choice", return_value="B")
    @patch("reber_dataset.random.randint", return_value=1)
    @patch("reber_dataset.EmbeddedReberGrammar")
    def test_split_and_fix_lengths_pads_to_max_length(self, embedded_cls, _randint, _choice):
        embedded_cls.return_value.produce.return_value = ["B", "T", "E"]

        dataset = ReberDataset(2)
        dataset._max_length = 5

        output = dataset.split_and_fix_lengths("A,B")

        self.assertEqual(output, ["A", "B", "<PAD>", "<PAD>", "<PAD>"])

    @patch("reber_dataset.random.choice", return_value="B")
    @patch("reber_dataset.random.randint", return_value=1)
    @patch("reber_dataset.EmbeddedReberGrammar")
    def test_split_and_fix_lengths_uses_custom_pad_value(self, embedded_cls, _randint, _choice):
        embedded_cls.return_value.produce.return_value = ["B", "T", "E"]

        dataset = ReberDataset(2, pad_value="<X>")
        dataset._max_length = 3

        output = dataset.split_and_fix_lengths("A")

        self.assertEqual(output, ["A", "<X>", "<X>"])

    @patch("reber_dataset.random.choice", return_value="B")
    @patch("reber_dataset.random.randint", return_value=1)
    @patch("reber_dataset.EmbeddedReberGrammar")
    def test_getitem_returns_fixed_length_sequences_for_all_indices(self, embedded_cls, _randint, _choice):
        embedded_cls.return_value.produce.return_value = ["B", "T", "E"]

        dataset = ReberDataset(6)
        expected_len = dataset._max_length

        for idx in range(len(dataset)):
            tokens, _ = dataset[idx]
            self.assertEqual(len(tokens), expected_len)

    @patch("reber_dataset.random.choice", return_value="B")
    @patch("reber_dataset.random.randint", return_value=1)
    @patch("reber_dataset.EmbeddedReberGrammar")
    def test_invalid_sequences_do_not_contain_empty_token(self, embedded_cls, _randint, _choice):
        embedded_cls.return_value.produce.return_value = ["B", "T", "E"]

        dataset = ReberDataset(4)

        invalid_tokens, invalid_label = dataset[len(dataset) // 2]
        decoded = [dataset.ID_TO_TOKEN[token_id.item()] for token_id in invalid_tokens]
        self.assertTrue(torch.equal(invalid_label, torch.tensor([0.0], dtype=torch.float32)))
        self.assertNotIn("", decoded)

    @patch("reber_dataset.random.choice", return_value="B")
    @patch("reber_dataset.random.randint", return_value=1)
    @patch("reber_dataset.EmbeddedReberGrammar")
    def test_max_length_is_based_on_token_count_not_string_length(self, embedded_cls, _randint, _choice):
        embedded_cls.return_value.produce.return_value = ["B", "T", "E"]

        dataset = ReberDataset(2)

        self.assertEqual(dataset._max_length, 3)

    @patch("reber_dataset.random.choice", return_value="B")
    @patch("reber_dataset.random.randint", return_value=1)
    @patch("reber_dataset.EmbeddedReberGrammar")
    def test_dataloader_can_collate_batch_without_runtime_error(self, embedded_cls, _randint, _choice):
        embedded_cls.return_value.produce.return_value = ["B", "T", "E"]

        dataset = ReberDataset(4)
        loader = DataLoader(dataset, batch_size=2)

        X_batch, y_batch = next(iter(loader))
        self.assertEqual(X_batch.shape, (2, dataset._max_length))
        self.assertEqual(y_batch.shape, (2, 1))
        self.assertEqual(X_batch.dtype, torch.long)
        self.assertEqual(y_batch.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
