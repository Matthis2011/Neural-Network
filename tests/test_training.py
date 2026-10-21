"""Tests for core neural-network behavior and training utilities."""

import argparse
import hashlib
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import model_manager_training as training
from model_manager import (
    ShellContext,
    TrainingJob,
    TrainingPlan,
    parse_duration,
)
from model_manager_models import (
    prompt_checkpoint_options,
    prompt_model_architecture,
    show_model_details,
)
from model_manager_shell import (
    get_interactive_training_options,
    handle_help,
    handle_history,
    handle_stats,
    handle_train,
    run_shell,
)
from model_manager_training import (
    get_target_epoch_estimate,
    load_data,
    parse_training_options,
    validate_training_data,
)
from neural_network import (
    LossReductionTarget,
    NeuralNetwork,
    TrainingSettings,
    argmax,
    softmax,
)


class TrainingUtilityTests(unittest.TestCase):
    """Check parsing, data loading, and basic model behavior."""

    def test_parse_duration_converts_supported_units(self):
        """Convert seconds, minutes, and fractional hours to seconds."""
        self.assertEqual(parse_duration("30s"), 30)
        self.assertEqual(parse_duration("10m"), 600)
        self.assertEqual(parse_duration("1.5h"), 5400)

    def test_parse_duration_rejects_invalid_values(self):
        """Reject durations without a supported positive unit."""
        for value in ("0s", "-1m", "3d", "ten minutes"):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                parse_duration(value)

    def test_parse_training_options_accepts_loss_targets(self):
        """Parse either absolute or percentage reduction goals."""
        absolute = parse_training_options(["--r=0.15", "--e=200", "--t=5m"])
        percentage_target = parse_training_options(
            ["--reduce-percent=20", "-v", "--b=16", "--lr=0.002"]
        )

        self.assertEqual(absolute.reduce, 0.15)
        self.assertEqual(absolute.epochs, 200)
        self.assertEqual(absolute.time, 300)
        self.assertIsNone(absolute.reduce_percent)
        self.assertEqual(percentage_target.reduce_percent, 20)
        self.assertEqual(percentage_target.batch_size, 16)
        self.assertEqual(percentage_target.learning_rate, 0.002)
        self.assertIsNone(percentage_target.reduce)
        self.assertTrue(percentage_target.verbose)

    def test_training_parser_accepts_shell_style_short_and_long_options(self):
        """Accept short flags, double-dash aliases, and descriptive long names."""
        short_options = parse_training_options(
            ["-e", "200", "-t", "5m", "-b", "16", "-lr", "0.002", "-y"]
        )
        long_options = parse_training_options(
            [
                "--epochs=200",
                "--time=5m",
                "--batch-size=16",
                "--learning-rate=0.002",
                "--yes",
            ]
        )

        self.assertEqual(short_options.epochs, long_options.epochs)
        self.assertEqual(short_options.time, long_options.time)
        self.assertEqual(short_options.batch_size, long_options.batch_size)
        self.assertEqual(short_options.learning_rate, long_options.learning_rate)
        self.assertTrue(short_options.yes)
        self.assertTrue(long_options.yes)

    def test_training_parser_accepts_validation_seed_and_checkpoint_options(self):
        """Parse validation, deterministic seed, checkpoint, and resume controls."""
        options = parse_training_options(
            [
                "--e=25",
                "--validation-split=0.2",
                "--patience=4",
                "--seed=0",
                "--checkpoint=state.json",
            ]
        )
        self.assertEqual(options.validation_split, 0.2)
        self.assertEqual(options.patience, 4)
        self.assertEqual(options.seed, 0)
        self.assertEqual(options.checkpoint, "state.json")

        resumed = parse_training_options(["--e=10", "--resume=state.json", "--seed=0"])
        self.assertEqual(resumed.resume, "state.json")
        self.assertEqual(resumed.seed, 0)

    def test_training_parser_rejects_invalid_validation_and_seed_values(self):
        """Reject impossible validation fractions and negative random seeds."""
        for arguments in (
            ["--e=10", "--validation-split=1"],
            ["--e=10", "--validation-split=nan"],
            ["--e=10", "--seed=-1"],
            ["--e=10", "--checkpoint="],
        ):
            with (
                self.subTest(arguments=arguments),
                patch("builtins.print"),
            ):
                self.assertIsNone(parse_training_options(arguments))

    def test_help_supports_concise_and_detailed_command_usage(self):
        """Keep default help concise and expose detailed topics on demand."""
        context = ShellContext("help", [], {}, None)
        output = StringIO()
        with redirect_stdout(output):
            handle_help(context)
        concise_help = output.getvalue()
        self.assertIn("help --<topic>", concise_help)
        self.assertNotIn("Loss targets can combine", concise_help)

        detailed_context = ShellContext("help", ["--train"], {}, None)
        output = StringIO()
        with redirect_stdout(output):
            handle_help(detailed_context)
        self.assertIn("-v, --verbose", output.getvalue())
        self.assertIn("Examples", output.getvalue())

        legacy_topic_context = ShellContext("help", ["-train"], {}, None)
        output = StringIO()
        with redirect_stdout(output):
            handle_help(legacy_topic_context)
        self.assertIn("Usage: help [--<topic>]", output.getvalue())

    def test_stats_accepts_detailed_and_short_flags(self):
        """Route both detailed stats spellings to the extended model display."""
        config = {"input_size": 2, "output_size": 2}
        selected = ("sample", config)
        for arguments in (["--detailed"], ["-d"]):
            with (
                self.subTest(arguments=arguments),
                patch("model_manager_shell.show_model_details") as show_details,
            ):
                handle_stats(
                    ShellContext("stats", arguments, {"sample": config}, selected)
                )
            show_details.assert_called_once_with("sample", config, detailed=True)

        output = StringIO()
        with redirect_stdout(output):
            handle_stats(
                ShellContext("stats", ["-detailed"], {"sample": config}, selected)
            )
        self.assertIn("Usage: stats [-d | --detailed]", output.getvalue())

    def test_detailed_stats_show_classes_and_latest_training_metrics(self):
        """Show class names and the useful results from the latest run."""
        config = {
            "input_size": 2,
            "output_size": 2,
            "hidden_sizes": [3],
            "input_scale": 1,
            "model_file": "sample.json",
            "data_file": "samples.json",
            "class_names": ["negative", "positive"],
        }
        latest_run = {
            "started_at": "2026-10-17T15:00:00+02:00",
            "epochs_completed": 12,
            "duration_seconds": 6.0,
            "seconds_per_epoch": 0.5,
            "dataset_size": 100,
            "batch_size": 8,
            "learning_rate": 0.01,
            "initial_loss": 0.9,
            "final_loss": 0.3,
            "loss_reduction": 0.6,
            "loss_reduction_percent": 66.67,
            "training_accuracy": 0.92,
            "validation_split": 0.2,
            "validation_loss": 0.25,
            "validation_accuracy": 0.94,
            "best_validation_epoch": 9,
            "seed": 42,
            "dataset_sha256": "a" * 64,
            "checkpoint_file": "checkpoints/sample.json",
            "target_loss_reduction_percent": 50,
            "target_reached": True,
        }
        output = StringIO()
        with (
            patch("model_manager_models.model_path_for", return_value=Path("sample")),
            patch("model_manager_models.data_path_for", return_value=Path("data")),
            patch(
                "model_manager_models.load_training_history",
                return_value={"sample": [latest_run]},
            ),
            redirect_stdout(output),
        ):
            show_model_details("sample", config, detailed=True)

        details = output.getvalue()
        for expected in (
            "negative",
            "positive",
            "Latest training run",
            "12",
            "Dataset size",
            "0.900000 -> 0.300000",
            "66.67%",
            "92.0%",
            "reached: yes",
            "Validation loss",
            "0.250000",
            "94.0%",
            "Best validation epoch",
            "Random seed",
            "42",
            "Checkpoint",
        ):
            self.assertIn(expected, details)

    def test_loss_targets_are_mutually_exclusive(self):
        """Reject a command that requests both forms of loss reduction."""
        with patch("builtins.print"):
            options = parse_training_options(["--r=0.15", "--reduce-percent=20"])

        self.assertIsNone(options)

    def test_target_epoch_estimate_uses_previous_loss_curve(self):
        """Estimate target epochs from matching historical training runs."""
        config = {"data_file": "samples.json"}
        history = {
            "sample": [
                {
                    "data_file": "samples.json",
                    "dataset_size": 4,
                    "batch_size": 2,
                    "learning_rate": 0.01,
                    "initial_loss": 1.0,
                    "final_loss": 0.6,
                    "epoch_losses": [1.0, 0.85, 0.7],
                    "epochs_completed": 3,
                }
            ]
        }
        with (
            patch("model_manager_training.load_data", return_value=[{}] * 4),
            patch(
                "model_manager_training.load_training_history",
                return_value=history,
            ),
        ):
            estimate, sample_count = get_target_epoch_estimate(
                "sample", config, 2, 0.01, LossReductionTarget(percentage=20)
            )

        self.assertEqual(estimate, 2)
        self.assertEqual(sample_count, 1)

    def test_target_epoch_estimate_supports_absolute_reduction(self):
        """Estimate absolute loss reduction from historical loss curves."""
        config = {"data_file": "samples.json"}
        history = {
            "sample": [
                {
                    "data_file": "samples.json",
                    "dataset_size": 4,
                    "batch_size": 2,
                    "learning_rate": 0.01,
                    "initial_loss": 1.0,
                    "final_loss": 0.6,
                    "epoch_losses": [1.0, 0.85, 0.7],
                    "epochs_completed": 3,
                }
            ]
        }
        with (
            patch("model_manager_training.load_data", return_value=[{}] * 4),
            patch(
                "model_manager_training.load_training_history",
                return_value=history,
            ),
        ):
            estimate, sample_count = get_target_epoch_estimate(
                "sample", config, 2, 0.01, LossReductionTarget(absolute=0.2)
            )

        self.assertEqual(estimate, 2)
        self.assertEqual(sample_count, 1)

    def test_target_command_uses_safety_cap_without_history(self):
        """Wire a target-only command to its finite fallback epoch limit."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "model.json"
            data_path = root / "data.json"
            model_path.touch()
            data_path.touch()
            config = {"model_file": "model.json", "data_file": "data.json"}
            context = ShellContext(
                "train",
                ["--reduce-percent=20", "-y", "-v"],
                {"sample": config},
                ("sample", config),
            )
            with (
                patch("model_manager_shell.model_path_for", return_value=model_path),
                patch("model_manager_shell.data_path_for", return_value=data_path),
                patch(
                    "model_manager_shell.get_epoch_time_estimate",
                    return_value=(None, 0),
                ),
                patch(
                    "model_manager_shell.get_target_epoch_estimate",
                    return_value=(None, 0),
                ),
                patch("model_manager_shell.continue_training") as continue_training,
                patch("builtins.print"),
            ):
                handle_train(context)

        plan = continue_training.call_args.args[2]
        self.assertEqual(plan.epoch_limit, training.MAX_TARGET_EPOCHS)
        self.assertEqual(plan.target_loss_reduction_percent, 20)
        self.assertTrue(plan.verbose)

    def test_interactive_train_reuses_saved_settings_and_accepts_overrides(self):
        """Use saved epochs by default while letting users override one value."""
        preferences = {
            "epochs": 40,
            "batch_size": 16,
            "learning_rate": 0.005,
        }
        with patch("builtins.input", return_value="--lr=0.01"):
            options = get_interactive_training_options("sample", preferences)

        self.assertEqual(options.epochs, 40)
        self.assertIsNone(options.batch_size)
        self.assertEqual(options.learning_rate, 0.01)

    def test_interactive_train_can_replace_saved_epoch_goal_with_time_budget(self):
        """Allow entering a different training objective from the saved default."""
        preferences = {
            "epochs": 40,
            "batch_size": 16,
            "learning_rate": 0.005,
        }
        with patch("builtins.input", return_value="--t=5m"):
            options = get_interactive_training_options("sample", preferences)

        self.assertIsNone(options.epochs)
        self.assertEqual(options.time, 300)

    def test_interactive_train_reprompts_after_invalid_changes(self):
        """Keep the setup open after a mistyped option so it can be corrected."""
        preferences = {
            "epochs": 40,
            "batch_size": 16,
            "learning_rate": 0.005,
        }
        with (
            patch("builtins.input", side_effect=["--lr=invalid", "--lr=0.01"]),
            redirect_stdout(StringIO()) as output,
        ):
            options = get_interactive_training_options("sample", preferences)

        self.assertEqual(options.epochs, 40)
        self.assertEqual(options.learning_rate, 0.01)
        self.assertIn("must be a number", output.getvalue())

    def test_interactive_train_requests_objective_when_no_epoch_default_exists(self):
        """Require a limit when a model has no previous epoch preference."""
        preferences = {
            "epochs": None,
            "batch_size": 32,
            "learning_rate": 0.001,
        }
        with patch("builtins.input", return_value="--e=25"):
            options = get_interactive_training_options("new model", preferences)

        self.assertEqual(options.epochs, 25)

    def test_train_without_arguments_previews_saved_plan_before_starting(self):
        """Route bare train through saved defaults, preview, and confirmation."""
        config = {"model_file": "sample.json", "data_file": "samples.json"}
        context = ShellContext("train", [], {"sample": config}, ("sample", config))
        preferences = {
            "epochs": 40,
            "batch_size": 16,
            "learning_rate": 0.005,
        }
        planned_run = object()
        with (
            patch("builtins.input", return_value=""),
            patch(
                "model_manager_shell.load_training_preferences",
                return_value=preferences,
            ),
            patch(
                "model_manager_shell.prepare_training_plan",
                return_value=planned_run,
            ) as prepare_plan,
            patch("model_manager_shell.confirm_training", return_value=True),
            patch("model_manager_shell.continue_training") as continue_training,
            redirect_stdout(StringIO()),
        ):
            handle_train(context)

        options = prepare_plan.call_args.args[2]
        self.assertEqual(options.epochs, 40)
        self.assertEqual(options.batch_size, None)
        self.assertEqual(options.learning_rate, None)
        continue_training.assert_called_once_with("sample", config, planned_run)

    def test_suggest_is_no_longer_a_shell_command(self):
        """Remove the standalone suggest command in favor of interactive train."""
        output = StringIO()
        with (
            patch("builtins.input", side_effect=["suggest", "exit"]),
            redirect_stdout(output),
        ):
            run_shell({})

        self.assertIn("Unknown command: suggest", output.getvalue())
        self.assertNotIn("suggest              ", output.getvalue())

    def test_history_displays_loss_target_result(self):
        """Show each run's target and whether it was reached."""
        config = {"data_file": "samples.json"}
        context = ShellContext("history", [], {"sample": config}, ("sample", config))
        history = {
            "sample": [
                {
                    "started_at": "2026-10-17T14:00:00+02:00",
                    "duration_seconds": 2,
                    "seconds_per_epoch": 1,
                    "epochs_completed": 2,
                    "batch_size": 2,
                    "learning_rate": 0.01,
                    "initial_loss": 1.0,
                    "final_loss": 0.8,
                    "loss_reduction": 0.2,
                    "loss_reduction_percent": 20,
                    "training_accuracy": 0.75,
                    "target_loss_reduction_percent": 15,
                    "target_reached": True,
                }
            ]
        }
        output = StringIO()
        with (
            patch(
                "model_manager_shell.load_training_history",
                return_value=history,
            ),
            redirect_stdout(output),
        ):
            handle_history(context)

        self.assertIn("target: loss reduction >= 15%", output.getvalue())
        self.assertIn("reached: yes", output.getvalue())

    def test_load_data_applies_configured_input_scale(self):
        """Load examples from data/ and scale only the inputs."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_directory = Path(temporary_directory) / "data"
            data_directory.mkdir()
            (data_directory / "samples.json").write_text(
                '[{"inputs": [10, 20], "targets": [1, 0]}]',
                encoding="utf-8",
            )

            with patch("model_manager_training.DATA_DIR", data_directory):
                examples = load_data(
                    {
                        "data_file": "samples.json",
                        "input_size": 2,
                        "output_size": 2,
                        "input_scale": 10,
                    }
                )

        self.assertEqual(examples, [{"inputs": [1.0, 2.0], "targets": [1, 0]}])

    def test_dataset_hash_uses_source_file_bytes(self):
        """Fingerprint the exact data file used for a manager training run."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_directory = Path(temporary_directory) / "data"
            data_directory.mkdir()
            data_file = data_directory / "samples.json"
            data_file.write_bytes(b'[{"inputs":[1],"targets":[1,0]}]')
            config = {"data_file": "samples.json"}
            expected_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()
            with patch("model_manager_training.DATA_DIR", data_directory):
                actual_hash = training.dataset_hash_for(config)

        self.assertEqual(actual_hash, expected_hash)

    def test_checkpoint_paths_cannot_overwrite_model_or_dataset(self):
        """Protect the files needed to save models and resume training."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            models = root / "models"
            data = root / "data"
            models.mkdir()
            data.mkdir()
            model_file = models / "model.json"
            data_file = data / "training.json"
            model_file.touch()
            data_file.touch()
            config = {
                "model_file": "model.json",
                "data_file": "training.json",
            }
            with (
                patch.object(training, "MODELS_DIR", models),
                patch.object(training, "DATA_DIR", data),
            ):
                for protected_file in (model_file, data_file):
                    with (
                        self.subTest(protected_file=protected_file.name),
                        self.assertRaisesRegex(ValueError, "must not overwrite"),
                    ):
                        training.validate_checkpoint_paths(config, (protected_file,))

    def test_create_prompt_explicitly_resumes_an_existing_checkpoint(self):
        """Ask before taking over an existing checkpoint in model creation."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "state.json"
            checkpoint.write_text("existing checkpoint", encoding="utf-8")
            with patch(
                "builtins.input",
                side_effect=[str(checkpoint), "yes"],
            ):
                save_path, resume_path = prompt_checkpoint_options(
                    {"model_file": "new-model.json", "data_file": "data.json"}
                )

        self.assertEqual(save_path, str(checkpoint))
        self.assertEqual(resume_path, str(checkpoint))

    def test_creation_architecture_prompt_returns_model_configuration(self):
        """Preserve architecture and class settings when extracting setup helpers."""
        with patch(
            "builtins.input",
            side_effect=["1", "8", "1", "model.json", "no", "yes"],
        ):
            config = prompt_model_architecture("training.json", 2, 2)

        self.assertEqual(config["data_file"], "training.json")
        self.assertEqual(config["model_file"], "model.json")
        self.assertEqual(config["hidden_sizes"], [8])
        self.assertEqual(config["class_names"], ["no", "yes"])

    def test_training_data_validation_accepts_valid_examples(self):
        """Accept finite, dimensionally correct one-hot examples."""
        data = [{"inputs": [0.25, 0.75], "targets": [0, 1]}]

        validate_training_data(
            data,
            {"input_size": 2, "output_size": 2},
            "samples.json",
        )

    def test_training_data_validation_keeps_circle_coordinate_shortcut(self):
        """Allow coordinate pairs for networks that derive the third feature."""
        validate_training_data(
            [{"inputs": [0.2, 0.3], "targets": [0, 1, 0]}],
            {"input_size": 3, "output_size": 3},
            "circle.json",
        )

    def test_training_data_validation_rejects_empty_data(self):
        """Fail early with a clear error when the dataset has no examples."""
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            validate_training_data(
                [], {"input_size": 1, "output_size": 2}, "empty.json"
            )

    def test_training_data_validation_reports_bad_example_and_dimensions(self):
        """Identify the exact row and expected input or target width."""
        cases = [
            ([{"inputs": [1], "targets": [1, 0]}], "example 1 has 1 inputs"),
            (
                [{"inputs": [1, 2], "targets": [1]}],
                "example 1 has 1 targets",
            ),
            (
                [{"inputs": [1, float("nan")], "targets": [1, 0]}],
                "input 2 must be a finite number",
            ),
            (
                [{"inputs": [1, 2], "targets": [1, float("inf")]}],
                "target 2 must be a finite number",
            ),
            (
                [{"inputs": [1, 2]}],
                "example 1 must contain input and target lists",
            ),
            (
                [None],
                "example 1 must be an object",
            ),
            (
                [{"inputs": [1, 2], "targets": [0, 0]}],
                "targets must be non-negative and sum to 1",
            ),
            (
                [{"inputs": [1, 2], "targets": [-0.1, 1.1]}],
                "targets must be non-negative and sum to 1",
            ),
        ]
        config = {"input_size": 2, "output_size": 2}
        for data, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_training_data(data, config, "samples.json")

    def test_training_returns_expected_metrics(self):
        """Train for the requested epochs and return timing/loss metrics."""
        examples = [
            {"inputs": [0.0, 0.0], "targets": [1.0, 0.0]},
            {"inputs": [1.0, 1.0], "targets": [0.0, 1.0]},
        ]
        network = NeuralNetwork(2, [3], 2)

        with patch("builtins.print"):
            metrics = network.train(examples, TrainingSettings(2, 1, 0.01))

        self.assertEqual(metrics["epochs_completed"], 2)
        self.assertEqual(len(metrics["epoch_losses"]), 2)
        self.assertEqual(len(metrics["epoch_durations_seconds"]), 2)
        self.assertEqual(metrics["dataset_size"], 2)
        self.assertGreaterEqual(metrics["training_accuracy"], 0)
        self.assertLessEqual(metrics["training_accuracy"], 1)

    def test_training_accuracy_is_not_divided_by_dataset_size_twice(self):
        """Report the correct-share fraction, not that fraction divided again."""
        examples = [
            {"inputs": [0.0, 0.0], "targets": [1.0, 0.0]},
            {"inputs": [1.0, 1.0], "targets": [0.0, 1.0]},
        ]
        network = NeuralNetwork(2, [3], 2, seed=5)

        with patch("builtins.print"):
            metrics = network.train(
                examples,
                TrainingSettings(epochs=1, batch_size=1, learning_rate=0.01),
            )

        correct = sum(
            argmax(softmax(network.forward(example["inputs"])))
            == argmax(example["targets"])
            for example in examples
        )
        self.assertEqual(metrics["training_accuracy"], correct / len(examples))

    def test_training_supports_networks_without_hidden_layers(self):
        """Train a direct input-to-output network without indexing hidden layers."""
        examples = [
            {"inputs": [0.0, 0.0], "targets": [1.0, 0.0]},
            {"inputs": [1.0, 1.0], "targets": [0.0, 1.0]},
        ]
        network = NeuralNetwork(2, [], 2)

        with patch("builtins.print"):
            metrics = network.train(examples, TrainingSettings(1, 1, 0.01))

        self.assertEqual(metrics["epochs_completed"], 1)
        self.assertEqual(metrics["dataset_size"], 2)

    def test_training_progress_is_compact_unless_verbose_is_enabled(self):
        """Avoid one output line per epoch by default, preserving a verbose mode."""
        examples = [
            {"inputs": [0.0, 0.0], "targets": [1.0, 0.0]},
            {"inputs": [1.0, 1.0], "targets": [0.0, 1.0]},
        ]
        network = NeuralNetwork(2, [3], 2)
        output = StringIO()
        with redirect_stdout(output):
            network.train(examples, TrainingSettings(2, 1, 0.01))
        self.assertNotIn("Epoch 1/2", output.getvalue())
        self.assertIn("Final loss:", output.getvalue())

        output = StringIO()
        with redirect_stdout(output):
            network.train(examples, TrainingSettings(2, 1, 0.01, verbose=True))
        self.assertIn("Epoch 1/2", output.getvalue())
        self.assertIn("Epoch 2/2", output.getvalue())

    def test_training_stops_when_reduction_target_is_reached(self):
        """Stop after an epoch once a requested tiny reduction is achieved."""
        examples = [
            {"inputs": [0.0, 0.0], "targets": [1.0, 0.0]},
            {"inputs": [1.0, 1.0], "targets": [0.0, 1.0]},
        ]
        random.seed(7)
        network = NeuralNetwork(2, [3], 2)

        with patch("builtins.print"):
            metrics = network.train(
                examples,
                TrainingSettings(
                    100,
                    1,
                    0.01,
                    loss_reduction_target=LossReductionTarget(percentage=1e-10),
                ),
            )

        self.assertTrue(metrics["target_reached"])
        self.assertEqual(metrics["epochs_completed"], 1)
        reduction_percent = (
            (metrics["initial_loss"] - metrics["final_loss"])
            / metrics["initial_loss"]
            * 100
        )
        self.assertGreaterEqual(reduction_percent, 1e-10)

    def test_training_respects_epoch_cap_before_target(self):
        """Report an unmet target when an explicit epoch limit is reached."""
        examples = [
            {"inputs": [0.0, 0.0], "targets": [1.0, 0.0]},
            {"inputs": [1.0, 1.0], "targets": [0.0, 1.0]},
        ]
        network = NeuralNetwork(2, [3], 2)

        with patch("builtins.print"):
            metrics = network.train(
                examples,
                TrainingSettings(
                    2,
                    1,
                    0.01,
                    loss_reduction_target=LossReductionTarget(absolute=100),
                ),
            )

        self.assertFalse(metrics["target_reached"])
        self.assertEqual(metrics["epochs_completed"], 2)

    def test_seed_reproduces_initialization_split_and_training(self):
        """Identical seeds should produce identical parameters and loss curves."""
        examples = [
            {"inputs": [0.0, 0.0], "targets": [1.0, 0.0]},
            {"inputs": [0.0, 1.0], "targets": [0.0, 1.0]},
            {"inputs": [1.0, 0.0], "targets": [0.0, 1.0]},
            {"inputs": [1.0, 1.0], "targets": [1.0, 0.0]},
        ]
        first = NeuralNetwork(2, [3], 2, seed=15)
        second = NeuralNetwork(2, [3], 2, seed=15)
        settings = TrainingSettings(
            epochs=4,
            batch_size=2,
            learning_rate=0.03,
            validation_split=0.25,
            seed=21,
        )

        with patch("builtins.print"):
            first_metrics = first.train(examples, settings)
            second_metrics = second.train(examples, settings)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first_metrics["epoch_losses"], second_metrics["epoch_losses"])
        self.assertEqual(
            first_metrics["validation_losses"], second_metrics["validation_losses"]
        )

    def test_validation_plateau_stops_and_restores_best_epoch(self):
        """Stop after the configured plateau and restore the best weights."""
        examples = [
            {"inputs": [0.0, 0.0], "targets": [1.0, 0.0]},
            {"inputs": [0.0, 1.0], "targets": [0.0, 1.0]},
            {"inputs": [1.0, 0.0], "targets": [0.0, 1.0]},
            {"inputs": [1.0, 1.0], "targets": [1.0, 0.0]},
        ]
        network = NeuralNetwork(2, [3], 2, seed=5)
        with (
            patch.object(network, "_train_epoch", return_value=0.8),
            patch.object(
                network,
                "_evaluate_loss",
                side_effect=[1.0, 0.5, 0.6, 0.7, 0.5],
            ),
            patch.object(network, "load_dict", wraps=network.load_dict) as restore,
            patch("builtins.print"),
        ):
            metrics = network.train(
                examples,
                TrainingSettings(
                    epochs=5,
                    batch_size=2,
                    learning_rate=0.01,
                    validation_split=0.5,
                    patience=1,
                    seed=9,
                ),
            )

        self.assertTrue(metrics["stopped_by_early_stopping"])
        self.assertEqual(metrics["epochs_completed"], 2)
        self.assertEqual(metrics["best_validation_epoch"], 1)
        self.assertEqual(metrics["validation_loss"], 0.5)
        restore.assert_called_once()

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        """Resume preserves split, RNG state, and exact epoch-boundary behavior."""
        examples = [
            {"inputs": [0.0, 0.0], "targets": [1.0, 0.0]},
            {"inputs": [0.0, 1.0], "targets": [0.0, 1.0]},
            {"inputs": [1.0, 0.0], "targets": [0.0, 1.0]},
            {"inputs": [1.0, 1.0], "targets": [1.0, 0.0]},
        ]
        settings = {
            "batch_size": 2,
            "learning_rate": 0.03,
            "validation_split": 0.25,
            "patience": 10,
            "seed": 21,
        }
        uninterrupted = NeuralNetwork(2, [3], 2, seed=15)
        interrupted = NeuralNetwork(2, [3], 2, seed=15)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "state.json"
            with patch("builtins.print"):
                uninterrupted.train(examples, TrainingSettings(epochs=4, **settings))
                interrupted.train(
                    examples,
                    TrainingSettings(
                        epochs=2, checkpoint_path=str(checkpoint), **settings
                    ),
                )
                resumed = NeuralNetwork(2, [3], 2, seed=99)
                metrics = resumed.train(
                    examples,
                    TrainingSettings(epochs=2, resume_from=str(checkpoint), **settings),
                )

        self.assertEqual(metrics["epochs_completed"], 4)
        self.assertEqual(uninterrupted.to_dict(), resumed.to_dict())

    def test_resume_rejects_dataset_changes(self):
        """Do not resume from state built for a different dataset."""
        examples = [
            {"inputs": [0.0], "targets": [1.0, 0.0]},
            {"inputs": [1.0], "targets": [0.0, 1.0]},
        ]
        changed_data = [*examples, {"inputs": [0.5], "targets": [1.0, 0.0]}]
        settings = {"batch_size": 1, "learning_rate": 0.01, "seed": 7}
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "state.json"
            with patch("builtins.print"):
                NeuralNetwork(1, [], 2, seed=3).train(
                    examples,
                    TrainingSettings(
                        epochs=1, checkpoint_path=str(checkpoint), **settings
                    ),
                )
                with self.assertRaisesRegex(ValueError, "dataset does not match"):
                    NeuralNetwork(1, [], 2).train(
                        changed_data,
                        TrainingSettings(
                            epochs=1, resume_from=str(checkpoint), **settings
                        ),
                    )

    def test_time_budget_caps_training_before_an_unmet_loss_target(self):
        """Stop at an epoch boundary when time expires before the loss target."""
        examples = [
            {"inputs": [0.0, 0.0], "targets": [1.0, 0.0]},
            {"inputs": [1.0, 1.0], "targets": [0.0, 1.0]},
        ]
        network = NeuralNetwork(2, [3], 2)

        with patch("builtins.print"):
            metrics = network.train(
                examples,
                TrainingSettings(
                    100,
                    1,
                    0.01,
                    max_duration=1e-12,
                    loss_reduction_target=LossReductionTarget(absolute=100),
                ),
            )

        self.assertTrue(metrics["stopped_by_time"])
        self.assertFalse(metrics["target_reached"])
        self.assertEqual(metrics["epochs_completed"], 1)

    def test_training_run_persists_weights_preferences_and_metrics(self):
        """Persist a completed run entirely within the configured directories."""
        examples = [
            {"inputs": [0.0, 0.0], "targets": [1.0, 0.0]},
            {"inputs": [1.0, 1.0], "targets": [0.0, 1.0]},
        ]
        network = NeuralNetwork(2, [3], 2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_directory = root / "config"
            models_directory = root / "models"
            config = {"model_file": "sample.json", "data_file": "sample.json"}
            job = TrainingJob(
                name="sample",
                config=config,
                network=network,
                training_data=examples,
                plan=TrainingPlan(
                    epoch_limit=1,
                    batch_size=1,
                    learning_rate=0.01,
                    requested_epochs=1,
                    validation_split=0.5,
                    seed=17,
                ),
            )
            with (
                patch.object(training, "CONFIG_DIR", config_directory),
                patch.object(
                    training,
                    "TRAINING_PREFERENCES_PATH",
                    config_directory / "training_preferences.json",
                ),
                patch.object(
                    training,
                    "LEGACY_TRAINING_PREFERENCES_PATH",
                    root / "legacy_preferences.json",
                ),
                patch.object(
                    training,
                    "TRAINING_HISTORY_PATH",
                    config_directory / "training_history.json",
                ),
                patch.object(
                    training,
                    "LEGACY_TRAINING_HISTORY_PATH",
                    root / "legacy_history.json",
                ),
                patch.object(training, "MODELS_DIR", models_directory),
                patch("builtins.print"),
            ):
                run = training.train_and_record(job)

            self.assertTrue((models_directory / "sample.json").is_file())
            self.assertTrue((config_directory / "training_preferences.json").is_file())
            self.assertTrue((config_directory / "training_history.json").is_file())
            self.assertEqual(run["epochs_completed"], 1)
            self.assertEqual(run["dataset_size"], 2)
            self.assertEqual(len(run["dataset_sha256"]), 64)
            self.assertEqual(run["seed"], 17)
            self.assertEqual(run["validation_split"], 0.5)
            self.assertEqual(run["training_dataset_size"], 1)
            self.assertEqual(run["validation_dataset_size"], 1)
            self.assertIsNotNone(run["validation_loss"])

    def test_model_save_and_load_preserves_predictions(self):
        """Round-trip learned weights without changing predictions."""
        network = NeuralNetwork(2, [3], 2)
        expected = softmax(network.forward([0.25, 0.75]))

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.json"
            network.save(model_path)
            loaded_network = NeuralNetwork.load(model_path)

        actual = softmax(loaded_network.forward([0.25, 0.75]))
        for expected_value, actual_value in zip(expected, actual, strict=True):
            self.assertAlmostEqual(expected_value, actual_value)

    def test_model_load_trims_unused_trailing_weights(self):
        """Load legacy models with extra weights ignored by the old forward pass."""
        network = NeuralNetwork(2, [3], 2, seed=4)
        model_data = network.to_dict()
        model_data["hidden_layers"][0][0]["weights"].extend([0.5, -0.5])

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "legacy-model.json"
            model_path.write_text(
                json.dumps(model_data),
                encoding="utf-8",
            )
            loaded_network = NeuralNetwork.load(model_path)

        self.assertEqual(len(loaded_network.hidden_layers[0].neurons[0].weights), 2)
        self.assertEqual(
            softmax(loaded_network.forward([0.25, 0.75])),
            softmax(network.forward([0.25, 0.75])),
        )

    def test_checkpoint_load_still_rejects_extra_neuron_weights(self):
        """Do not relax strict dimensions when restoring a training checkpoint."""
        network = NeuralNetwork(2, [3], 2, seed=4)
        checkpoint_data = network.to_dict()
        checkpoint_data["hidden_layers"][0][0]["weights"].append(0.5)

        with self.assertRaisesRegex(ValueError, "neuron dimensions"):
            network.load_dict(checkpoint_data)


if __name__ == "__main__":
    unittest.main()
