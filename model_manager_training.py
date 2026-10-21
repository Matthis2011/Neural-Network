"""Training configuration, persistence, and estimation helpers."""

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Callable

from neural_network import (
    CIRCLE_COORDINATE_COUNT,
    CIRCLE_FEATURE_INPUT_SIZE,
    LossReductionTarget,
    NeuralNetwork,
    TrainingSettings,
)
from terminal_colors import colorize

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

MODELS_DIR = BASE_DIR / "models"

CONFIG_DIR = BASE_DIR / "config"

CONFIG_PATH = CONFIG_DIR / "model_configs.json"

TRAINING_PREFERENCES_PATH = CONFIG_DIR / "training_preferences.json"

TRAINING_HISTORY_PATH = CONFIG_DIR / "training_history.json"

LEGACY_CONFIG_PATH = BASE_DIR / "model_configs.json"

LEGACY_TRAINING_PREFERENCES_PATH = BASE_DIR / "training_preferences.json"

LEGACY_TRAINING_HISTORY_PATH = BASE_DIR / "training_history.json"

MAX_TIME_BUDGET_EPOCHS = 1_000_000

MAX_TARGET_EPOCHS = 1_000

TARGET_EPOCH_ESTIMATE_MULTIPLIER = 2

SECONDS_PER_MINUTE = 60

SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
PERCENTAGE_SCALE = 100

DEFAULT_TRAINING_PREFERENCES = {
    "epochs": None,
    "batch_size": 32,
    "learning_rate": 0.001,
}

ModelConfig = dict[str, Any]

ModelConfigs = dict[str, ModelConfig]

ModelSelection = tuple[str, ModelConfig]


@dataclass(frozen=True)
class ShellContext:
    """Hold the parsed command and current model state for a shell handler."""

    command: str
    arguments: list[str]
    configs: ModelConfigs
    selected: ModelSelection | None


@dataclass(frozen=True)
class TrainingPlan:
    """Capture the requested and resolved settings for one training run."""

    epoch_limit: int
    batch_size: int
    learning_rate: float
    time_budget: float | None = None
    requested_epochs: int | None = None
    target_loss_reduction: float | None = None
    target_loss_reduction_percent: float | None = None
    verbose: bool = False
    validation_split: float = 0.0
    patience: int = 10
    seed: int | None = None
    checkpoint_path: str | None = None
    resume_from: str | None = None


@dataclass
class TrainingJob:
    """Bundle a model, dataset, and plan for training and metric recording."""

    name: str
    config: ModelConfig
    network: NeuralNetwork
    training_data: list[dict[str, Any]]
    plan: TrainingPlan
    dataset_hash: str | None = None


CommandHandler = Callable[[ShellContext], ModelSelection | None]


DEFAULT_CONFIGS = {
    "Kreis-Klassifikation": {
        "data_file": "circle_point_classification_training.json",
        "model_file": "circle_point_classification_model.json",
        "input_size": 3,
        "output_size": 3,
        "hidden_sizes": [8, 8],
        "input_scale": 1,
        "class_names": ["Klasse 0", "Klasse 1", "Klasse 2"],
    },
    "Primzahl-Klassifikation": {
        "data_file": "prime_training.json",
        "model_file": "prime_model.json",
        "input_size": 1,
        "output_size": 2,
        "hidden_sizes": [16, 16],
        "input_scale": 9999,
        "class_names": ["keine Primzahl", "Primzahl"],
    },
    "Odd-Even-Diagnose-Binär": {
        "data_file": "odd_even_binary_training.json",
        "model_file": "odd_even_binary_model.json",
        "input_size": 10,
        "output_size": 2,
        "hidden_sizes": [16, 16],
        "input_scale": 1,
        "input_mode": "binary",
        "class_names": ["ungerade", "gerade"],
    },
    "Tic-Tac-Toe-KI": {
        "data_file": "tic_tac_toe_training_data.json",
        "model_file": "tic_tac_toe_model.json",
        "input_size": 9,
        "output_size": 9,
        "hidden_sizes": [32, 32],
        "input_scale": 1,
        "input_mode": "tic_tac_toe",
        "class_names": [f"Feld {index}" for index in range(9)],
    },
}


def load_configs():
    """Load model configurations and add any missing built-in models."""
    config_path = CONFIG_PATH if CONFIG_PATH.exists() else LEGACY_CONFIG_PATH
    if not config_path.exists():
        return dict(DEFAULT_CONFIGS)
    with config_path.open("r", encoding="utf-8") as file:
        configs = json.load(file)
    for name, config in DEFAULT_CONFIGS.items():
        configs.setdefault(name, config)
    return configs


def save_configs(configs):
    """Persist model configurations to the manager's JSON file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(configs, file, indent=4)


def resolve_asset_path(filename, directory):
    """Resolve an asset in its organized directory, then try its legacy path."""
    path = Path(filename)
    if path.is_absolute():
        return path
    organized_path = directory / path
    legacy_path = BASE_DIR / path
    if organized_path.exists() or not legacy_path.exists():
        return organized_path
    return legacy_path


def data_path_for(config):
    """Resolve the training-data path from a model configuration."""
    return resolve_asset_path(config["data_file"], DATA_DIR)


def model_path_for(config):
    """Resolve the model-file path from a model configuration."""
    return resolve_asset_path(config["model_file"], MODELS_DIR)


def validate_checkpoint_paths(config, filenames):
    """Prevent checkpoint output from replacing model, data, or manager state."""
    protected_paths = {
        path.resolve()
        for path in (
            model_path_for(config),
            data_path_for(config),
            CONFIG_PATH,
            LEGACY_CONFIG_PATH,
            TRAINING_HISTORY_PATH,
            LEGACY_TRAINING_HISTORY_PATH,
            TRAINING_PREFERENCES_PATH,
            LEGACY_TRAINING_PREFERENCES_PATH,
        )
    }
    for filename in filenames:
        if filename and Path(filename).expanduser().resolve() in protected_paths:
            raise ValueError(
                "Checkpoint files must not overwrite the model, dataset, or "
                "training configuration/history files."
            )


def load_data(config):
    """Load training examples and apply the model's configured input scale."""
    path = data_path_for(config)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    validate_training_data(data, config, path)
    input_scale = config.get("input_scale", 1)
    if input_scale != 1:
        for item in data:
            item["inputs"] = [value / input_scale for value in item["inputs"]]
    return data


def dataset_hash_for(config):
    """Return a SHA-256 identity for the exact source dataset file."""
    return hashlib.sha256(data_path_for(config).read_bytes()).hexdigest()


def expected_input_sizes(configured_size):
    """Return accepted input lengths, including the circle feature shortcut."""
    if configured_size == CIRCLE_FEATURE_INPUT_SIZE:
        return CIRCLE_FEATURE_INPUT_SIZE, CIRCLE_COORDINATE_COUNT
    return (configured_size,)


def validate_training_data(data, config, source="training data"):
    """Reject empty, malformed, non-finite, or dimensionally incompatible examples."""
    if not isinstance(data, list) or not data:
        raise ValueError(f"{source}: expected a non-empty list of examples.")
    expected_inputs = config["input_size"]
    expected_targets = config["output_size"]
    if (
        not isinstance(expected_inputs, int)
        or isinstance(expected_inputs, bool)
        or expected_inputs < 1
        or not isinstance(expected_targets, int)
        or isinstance(expected_targets, bool)
        or expected_targets < 1
    ):
        raise ValueError(f"{source}: model input and output sizes must be positive.")
    for row_number, example in enumerate(data, 1):
        row = f"{source}: example {row_number}"
        if not isinstance(example, dict):
            raise ValueError(f"{row} must be an object with inputs and targets.")
        inputs = example.get("inputs")
        targets = example.get("targets")
        if not isinstance(inputs, list) or not isinstance(targets, list):
            raise ValueError(f"{row} must contain input and target lists.")
        if len(inputs) not in expected_input_sizes(expected_inputs):
            expected = (
                f"{expected_inputs} (or {CIRCLE_COORDINATE_COUNT} coordinates)"
                if expected_inputs == CIRCLE_FEATURE_INPUT_SIZE
                else str(expected_inputs)
            )
            raise ValueError(f"{row} has {len(inputs)} inputs; expected {expected}.")
        if len(targets) != expected_targets:
            raise ValueError(
                f"{row} has {len(targets)} targets; expected {expected_targets}."
            )
        for field_name, values in (("input", inputs), ("target", targets)):
            for value_index, value in enumerate(values, 1):
                try:
                    valid_number = (
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and math.isfinite(value)
                    )
                except OverflowError:
                    valid_number = False
                if not valid_number:
                    raise ValueError(
                        f"{row} {field_name} {value_index} must be a finite number."
                    )
        if any(value < 0 for value in targets) or not math.isclose(
            sum(targets), 1.0, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(f"{row} targets must be non-negative and sum to 1.")


def ask_int(prompt, minimum=1):
    """Prompt until the user enters an integer at or above the minimum."""
    while True:
        try:
            value = int(input(colorize(prompt, "prompt")))
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(
            colorize(
                f"Enter a whole number greater than or equal to {minimum}.",
                "error",
            )
        )


def ask_float(prompt, minimum=None):
    """Prompt until the user enters a finite float meeting the minimum."""
    while True:
        try:
            value = float(input(colorize(prompt, "prompt")))
            if math.isfinite(value) and (minimum is None or value >= minimum):
                return value
        except ValueError:
            pass
        print(colorize("Enter a valid finite number.", "error"))


class TrainingArgumentParser(argparse.ArgumentParser):
    """Argument parser that reports command errors without exiting the shell."""

    def error(self, message):
        """Raise parse errors so the command loop can continue."""
        raise ValueError(message)


def positive_int(value):
    """Parse an integer argument that must be greater than zero."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a whole number") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value):
    """Parse a finite floating-point argument that must be greater than zero."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def non_negative_int(value):
    """Parse an integer seed, including zero."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a whole number") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def validation_fraction(value):
    """Parse a validation fraction where zero disables validation."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return parsed


def checkpoint_filename(value):
    """Require a non-empty file path for checkpoint and resume options."""
    if not value.strip():
        raise argparse.ArgumentTypeError("file path must not be empty")
    return value


def percentage(value):
    """Parse a finite percentage in the range greater than zero to 100."""
    parsed = positive_float(value)
    if parsed > PERCENTAGE_SCALE:
        raise argparse.ArgumentTypeError("must not exceed 100 percent")
    return parsed


def parse_duration(value):
    """Convert a time budget such as 30s, 10m, or 1.5h to seconds."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh])", value.lower())
    if match is None:
        raise argparse.ArgumentTypeError("use a duration such as 30s, 10m, or 1.5h")
    amount = float(match.group(1))
    multiplier = {"s": 1, "m": SECONDS_PER_MINUTE, "h": SECONDS_PER_HOUR}[
        match.group(2)
    ]
    duration = amount * multiplier
    if not math.isfinite(duration) or duration <= 0:
        raise argparse.ArgumentTypeError("duration must be greater than zero")
    return duration


def parse_training_options(arguments):
    """Parse fixed-epoch and time-budget training options."""
    parser = TrainingArgumentParser(
        prog="train",
        description="Continue training the selected model.",
        add_help=False,
    )
    parser.add_argument("-e", "--e", "--epochs", dest="epochs", type=positive_int)
    parser.add_argument(
        "-b", "--b", "--batch-size", dest="batch_size", type=positive_int
    )
    parser.add_argument(
        "-lr", "--lr", "--learning-rate", dest="learning_rate", type=positive_float
    )
    parser.add_argument("-t", "--t", "--time", dest="time", type=parse_duration)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "-r", "--r", "--reduce", dest="reduce", type=positive_float
    )
    target_group.add_argument(
        "--reduce-percent", dest="reduce_percent", type=percentage
    )
    parser.add_argument("-y", "--yes", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--validation-split", type=validation_fraction, default=0.0)
    parser.add_argument("--patience", type=positive_int, default=10)
    parser.add_argument("--seed", type=non_negative_int)
    parser.add_argument("--checkpoint", type=checkpoint_filename)
    parser.add_argument("--resume", type=checkpoint_filename)
    try:
        options = parser.parse_args(arguments)
        if (
            options.epochs is None
            and options.time is None
            and options.reduce is None
            and options.reduce_percent is None
        ):
            raise ValueError("provide an epoch count, time budget, or loss target")
    except ValueError as error:
        print(
            colorize(
                f"{error}\nUsage: train [-e <epochs> | --e=<epochs>] "
                "[-t <duration> | --t=<duration>] "
                "[-r <loss-reduction> | --r=<loss-reduction> | "
                "--reduce-percent=<percent>] "
                "[-b <batch-size> | --b=<batch-size>] "
                "[-lr <rate> | --lr=<rate>] [--validation-split=<fraction>] "
                "[--patience=<epochs>] [--seed=<integer>] "
                "[--checkpoint=<file> | --resume=<file>] [-y] [-v]",
                "error",
            )
        )
        return None
    return options


def load_training_preferences(name):
    """Load the last successful training options for one model."""
    preferences_path = (
        TRAINING_PREFERENCES_PATH
        if TRAINING_PREFERENCES_PATH.exists()
        else LEGACY_TRAINING_PREFERENCES_PATH
    )
    if not preferences_path.exists():
        return DEFAULT_TRAINING_PREFERENCES.copy()
    with preferences_path.open("r", encoding="utf-8") as file:
        preferences = json.load(file)
    model_preferences = preferences.get(name, {})
    return {
        "epochs": model_preferences.get("epochs"),
        "batch_size": model_preferences.get(
            "batch_size", DEFAULT_TRAINING_PREFERENCES["batch_size"]
        ),
        "learning_rate": model_preferences.get(
            "learning_rate", DEFAULT_TRAINING_PREFERENCES["learning_rate"]
        ),
    }


def save_training_preferences(name, epochs, batch_size, learning_rate):
    """Remember the latest successful training options for a model."""
    preferences_path = (
        TRAINING_PREFERENCES_PATH
        if TRAINING_PREFERENCES_PATH.exists()
        else LEGACY_TRAINING_PREFERENCES_PATH
    )
    if preferences_path.exists():
        with preferences_path.open("r", encoding="utf-8") as file:
            preferences = json.load(file)
    else:
        preferences = {}
    preferences[name] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with TRAINING_PREFERENCES_PATH.open("w", encoding="utf-8") as file:
        json.dump(preferences, file, indent=4)


def load_training_history():
    """Load the complete training history keyed by model name."""
    history_path = (
        TRAINING_HISTORY_PATH
        if TRAINING_HISTORY_PATH.exists()
        else LEGACY_TRAINING_HISTORY_PATH
    )
    if not history_path.exists():
        return {}
    with history_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_training_run(name, run):
    """Append one completed run's metrics to the persistent history."""
    history = load_training_history()
    history.setdefault(name, []).append(run)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with TRAINING_HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def get_evolutionary_time_estimate(name, architecture):
    """Return the median duration per generation from recent comparable runs."""
    comparable_runs = [
        run
        for run in load_training_history().get(name, [])
        if run.get("training_method") == "evolutionary"
        and run.get("architecture") == architecture
        and (run.get("seconds_per_generation") or 0) > 0
    ]
    recent_times = [run["seconds_per_generation"] for run in comparable_runs[-10:]]
    if not recent_times:
        return None, 0
    return median(recent_times), len(recent_times)


def get_epoch_time_estimate(name, config, batch_size):
    """Return median seconds per epoch from recent comparable training runs."""
    dataset_path = config["data_file"]
    dataset_size = len(load_data(config))
    comparable_runs = [
        run
        for run in load_training_history().get(name, [])
        if run.get("data_file") == dataset_path
        and run.get("dataset_size") == dataset_size
        and run.get("batch_size") == batch_size
        and (run.get("seconds_per_epoch") or 0) > 0
    ]
    recent_times = [run["seconds_per_epoch"] for run in comparable_runs[-10:]]
    if not recent_times:
        return None, 0
    return median(recent_times), len(recent_times)


def get_target_epoch_estimate(
    name,
    config,
    batch_size,
    learning_rate,
    target: LossReductionTarget,
):
    """Estimate epochs to a loss target from recent comparable training curves."""
    dataset_size = len(load_data(config))
    matching_runs = [
        run
        for run in load_training_history().get(name, [])
        if run.get("data_file") == config["data_file"]
        and run.get("dataset_size") == dataset_size
        and run.get("batch_size") == batch_size
        and math.isclose(run.get("learning_rate", -1), learning_rate)
        and not run.get("validation_split", 0)
    ]
    epoch_estimates = []
    for run in matching_runs[-10:]:
        initial_loss = run.get("initial_loss")
        final_loss = run.get("final_loss")
        epoch_losses = run.get("epoch_losses", [])
        epochs_completed = run.get("epochs_completed", 0)
        if (
            not isinstance(initial_loss, (int, float))
            or not isinstance(final_loss, (int, float))
            or epochs_completed < 1
            or len(epoch_losses) < epochs_completed
        ):
            continue
        achieved_reduction = initial_loss - final_loss
        achieved_percent = (
            achieved_reduction / initial_loss * 100 if initial_loss else 0
        )
        if target.absolute is not None and achieved_reduction < target.absolute:
            continue
        if target.percentage is not None and achieved_percent < target.percentage:
            continue

        loss_curve = [initial_loss, *epoch_losses[1:epochs_completed], final_loss]
        for epoch_count, measured_loss in enumerate(loss_curve[1:], 1):
            reduction = initial_loss - measured_loss
            reduction_percent = (
                reduction / initial_loss * PERCENTAGE_SCALE if initial_loss else 0
            )
            reached = (
                target.absolute is not None and reduction >= target.absolute
            ) or (
                target.percentage is not None and reduction_percent >= target.percentage
            )
            if reached:
                epoch_estimates.append(epoch_count)
                break

    if not epoch_estimates:
        return None, 0
    return median(epoch_estimates), len(epoch_estimates)


def estimate_epochs_for_budget(time_budget, seconds_per_epoch):
    """Estimate whole epochs within a budget, reserving time for overhead."""
    if seconds_per_epoch is None:
        return None
    return max(1, int(time_budget * 0.9 / seconds_per_epoch))


def train_and_record(job: TrainingJob):
    """Train a model, save it, and persist all available run metrics."""
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started_clock = time.perf_counter()
    dataset_hash = (
        job.dataset_hash
        or hashlib.sha256(
            json.dumps(
                job.training_data,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    metrics = job.network.train(
        job.training_data,
        TrainingSettings(
            epochs=job.plan.epoch_limit,
            batch_size=job.plan.batch_size,
            learning_rate=job.plan.learning_rate,
            max_duration=job.plan.time_budget,
            loss_reduction_target=LossReductionTarget(
                absolute=job.plan.target_loss_reduction,
                percentage=job.plan.target_loss_reduction_percent,
            ),
            verbose=job.plan.verbose,
            validation_split=job.plan.validation_split,
            patience=job.plan.patience,
            seed=job.plan.seed,
            checkpoint_path=job.plan.checkpoint_path or job.plan.resume_from,
            resume_from=job.plan.resume_from,
            dataset_hash=dataset_hash,
        ),
    )
    model_path = model_path_for(job.config)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    job.network.save(model_path)
    duration = time.perf_counter() - started_clock
    initial_loss = metrics["initial_loss"]
    final_loss = metrics["final_loss"]
    loss_reduction = initial_loss - final_loss
    run = {
        "started_at": started_at,
        "model_file": job.config["model_file"],
        "data_file": job.config["data_file"],
        "architecture": {
            "input_size": job.network.input_size,
            "hidden_sizes": job.network.hidden_sizes,
            "output_size": job.network.output_size,
        },
        "epochs_requested": job.plan.requested_epochs,
        "epoch_limit": job.plan.epoch_limit,
        "epochs_completed": metrics["epochs_completed"],
        "batch_size": job.plan.batch_size,
        "learning_rate": job.plan.learning_rate,
        "time_budget_seconds": job.plan.time_budget,
        "time_budget_reached": metrics["stopped_by_time"],
        "target_loss_reduction": job.plan.target_loss_reduction,
        "target_loss_reduction_percent": job.plan.target_loss_reduction_percent,
        "target_reached": metrics["target_reached"],
        "epoch_limit_reached": (
            not metrics["target_reached"]
            and not metrics["stopped_by_time"]
            and not metrics["stopped_by_early_stopping"]
            and metrics["epochs_completed"] >= job.plan.epoch_limit
        ),
        "duration_seconds": duration,
        "seconds_per_epoch": (
            sum(metrics["epoch_durations_seconds"]) / metrics["epochs_completed"]
            if metrics["epochs_completed"]
            else None
        ),
        "epoch_durations_seconds": metrics["epoch_durations_seconds"],
        "epoch_losses": metrics["epoch_losses"],
        "dataset_size": metrics["dataset_size"],
        "training_dataset_size": metrics["training_dataset_size"],
        "validation_dataset_size": metrics["validation_dataset_size"],
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": loss_reduction,
        "loss_reduction_percent": (
            loss_reduction / initial_loss * 100 if initial_loss else None
        ),
        "training_accuracy": metrics["training_accuracy"],
        "validation_split": job.plan.validation_split,
        "validation_losses": metrics["validation_losses"],
        "validation_loss": metrics["validation_loss"],
        "validation_accuracy": metrics["validation_accuracy"],
        "best_validation_loss": metrics["best_validation_loss"],
        "best_validation_epoch": metrics["best_validation_epoch"],
        "stopped_by_early_stopping": metrics["stopped_by_early_stopping"],
        "seed": job.plan.seed,
        "dataset_sha256": dataset_hash,
        "checkpoint_file": job.plan.checkpoint_path or job.plan.resume_from,
    }
    save_training_preferences(
        job.name,
        metrics["epochs_completed"],
        job.plan.batch_size,
        job.plan.learning_rate,
    )
    save_training_run(job.name, run)
    return run


def format_duration(seconds):
    """Format a duration in seconds using a compact human-readable unit."""
    if seconds >= SECONDS_PER_HOUR:
        return f"{seconds / SECONDS_PER_HOUR:.1f}h"
    if seconds >= SECONDS_PER_MINUTE:
        return f"{seconds / SECONDS_PER_MINUTE:.1f}m"
    return f"{seconds:.1f}s"


def confirm_training():
    """Ask for confirmation before starting a training run."""
    try:
        answer = input(colorize("Start training? [y/N] ", "warning")).strip().lower()
    except EOFError:
        print(colorize("\nTraining cancelled.", "warning"))
        return False
    if answer not in {"y", "yes"}:
        print(colorize("Training cancelled.", "warning"))
        return False
    return True
