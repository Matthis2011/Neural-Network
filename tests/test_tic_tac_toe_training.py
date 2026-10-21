"""Tests for generation of valid Tic-Tac-Toe training targets."""

import unittest

from scripts.tic_tac_toe_training import build_target_vector


class TicTacToeTrainingTests(unittest.TestCase):
    """Validate generated target vectors."""

    def test_best_moves_share_probability_equally(self):
        """Assign equal probability only to the optimal moves."""
        target = build_target_vector([0, 8])

        self.assertEqual(len(target), 9)
        self.assertEqual(target[0], 0.5)
        self.assertEqual(target[8], 0.5)
        self.assertEqual(sum(target), 1)

    def test_target_requires_at_least_one_move(self):
        """Reject empty best-move lists rather than returning invalid labels."""
        with self.assertRaises(ValueError):
            build_target_vector([])


if __name__ == "__main__":
    unittest.main()
