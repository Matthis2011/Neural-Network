"""Tests for evolutionary Connect Four progress reporting."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

import connect_four


class ConnectFourProgressTests(unittest.TestCase):
    """Test progress reporting without running real training games."""

    @patch("connect_four.play_game", return_value=1)
    def test_ai_game_loop_reports_each_completed_game(self, play_game):
        """Report progress after each game without changing fitness."""
        population = [object() for _ in range(4)]
        fitnesses = [0, 0, 0, 0]
        progress_callback = Mock()

        connect_four.ai_game_loop(
            population,
            population_size=4,
            pool=[0, 1, 2, 3],
            first_player=1,
            fitnesses=fitnesses,
            progress_callback=progress_callback,
        )

        self.assertEqual(play_game.call_count, 2)
        self.assertEqual(fitnesses, [1, 0, 1, 0])
        self.assertEqual(progress_callback.call_count, 2)
        progress_callback.assert_called_with(1)

    @patch("connect_four.play_game", return_value=0)
    def test_ai_game_loop_remains_compatible_without_callback(self, play_game):
        """Keep existing calls without a progress callback working."""
        population = [object() for _ in range(4)]
        fitnesses = [0, 0, 0, 0]

        connect_four.ai_game_loop(
            population,
            population_size=4,
            pool=[0, 1, 2, 3],
            first_player=1,
            fitnesses=fitnesses,
        )

        self.assertEqual(play_game.call_count, 2)
        self.assertEqual(fitnesses, [0.5, 0.5, 0.5, 0.5])

    @patch("connect_four.play_game", return_value=0)
    def test_select_parents_reports_all_1000_games(self, play_game):
        """Report all games across both player orders and ten rounds."""
        population = [object() for _ in range(100)]
        progress_callback = Mock()

        top_ten, new_population, returned_population = (
            connect_four.select_parents(
                population,
                progress_callback=progress_callback,
            )
        )

        self.assertEqual(play_game.call_count, 1000)
        self.assertEqual(progress_callback.call_count, 1000)
        self.assertEqual(len(top_ten), 10)
        self.assertEqual(len(new_population), 2)
        self.assertIs(returned_population, population)

    def test_evolutionary_training_reports_generations_and_final_evaluation(self):
        """Count all games and display the final evaluation phase."""
        population = [object() for _ in range(100)]
        progress_calls = []

        def fake_select_parents(current_population, progress_callback=None):
            for _ in range(1000):
                progress_callback(1)
                progress_calls.append(1)
            return list(range(10)), [object(), object()], population

        output = StringIO()
        with (
            patch("connect_four.read_parameters", return_value=population),
            patch(
                "connect_four.select_parents",
                side_effect=fake_select_parents,
            ) as select_parents,
            patch("connect_four.create_child", return_value=object()),
            patch("connect_four.save_parameters") as save_parameters,
            redirect_stdout(output),
        ):
            connect_four.evolutionary_training(
                generations=1,
                model_path="unused-test-model.json",
            )

        self.assertEqual(select_parents.call_count, 2)
        self.assertEqual(len(progress_calls), 2000)
        self.assertTrue(save_parameters.called)

        displayed = output.getvalue()
        self.assertIn("Generation 1/1", displayed)
        self.assertIn("Abschließende Bewertung", displayed)
        self.assertIn("Spiele: 2000/2000", displayed)
        self.assertIn("ETA:", displayed)


if __name__ == "__main__":
    unittest.main()