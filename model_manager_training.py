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
"""Interactive command handlers and shell loop for the model manager."""

import argparse
import math
import shlex
import textwrap
from dataclasses import dataclass

from model_manager_models import (
    continue_training,
    create_new_model,
    print_model_list,
    resolve_model,
    show_model_details,
    use_model,
)
from model_manager_training import (
    MAX_TARGET_EPOCHS,
    MAX_TIME_BUDGET_EPOCHS,
    TARGET_EPOCH_ESTIMATE_MULTIPLIER,
    CommandHandler,
    ModelConfigs,
    ModelSelection,
    ShellContext,
    TrainingPlan,
    confirm_training,
    data_path_for,
    estimate_epochs_for_budget,
    format_duration,
    get_epoch_time_estimate,
    get_target_epoch_estimate,
    load_configs,
    load_training_history,
    load_training_preferences,
    model_path_for,
    parse_training_options,
    positive_int,
    validate_checkpoint_paths,
)
from neural_network import LossReductionTarget
from terminal_colors import colorize


def print_help_text(text):
    """Print help with its heading highlighted and usage text left readable."""
    lines = textwrap.dedent(text).strip().splitlines()
    if lines:
        print(colorize(lines[0], "header", bold=True))
        print("\n".join(lines[1:]))


@dataclass(frozen=True)
class TrainingPreview:
    """Collect the estimates and resolved limits shown before training."""

    epoch_time: float | None
    comparable_runs: int
    estimated_epochs: int | None
    target_epochs: float | None
    target_runs: int
    epoch_limit: int
    has_target: bool


def print_help(topic=None):
    """Print concise shell help or detailed usage for one command."""
    topics = {
        "train": """
Training
  train                             Interactively review and adjust saved settings
  train -e <n>, --e=<n>             Train for a fixed number of epochs
  train -t <time>, --t=<time>       Train up to a time budget (e.g. 10m, 1h)
  train -r <loss>, --r=<loss>       Stop after reducing loss by this amount
  train --reduce-percent=<pct>      Stop after reducing loss by this percentage
  -b <size>, --b=<size>             Batch size (defaults to this model's last run)
  -lr <rate>, --lr=<rate>           Learning rate (defaults to this model's last run)
  -y, --yes                         Skip the confirmation prompt
  -v, --verbose                     Print loss for every epoch
  --validation-split=<fraction>     Hold out data and enable early stopping
  --patience=<epochs>               Stop after this many unimproved epochs (10)
  --seed=<integer>                  Reproduce the data split and epoch shuffling
  --checkpoint=<file>               Save a resumable checkpoint after each epoch
  --resume=<file>                   Resume a compatible checkpoint

Long option names are also available, such as --epochs=500 and --time=10m.
Validation is opt-in; for example, --validation-split=0.2 reserves 20% of the
data and restores the weights from the best validation-loss epoch.
Training data is checked for finite numeric values, matching dimensions, and
non-negative targets that sum to 1 before training starts.
Checkpointing is also opt-in. --resume updates the same checkpoint unless a
different --checkpoint file is supplied; --e specifies additional epochs.
Resume with the same batch size, learning rate, validation split, patience,
seed, loss target, and unchanged dataset.
Loss targets can combine with -e/--e and/or -t/--t; limits stop training first.
Time budgets are checked at epoch boundaries. Comparable run history estimates
the duration or target epoch count; without it, timed runs learn from the first
epoch and target-only runs use a 1,000-epoch safety limit.

Examples
  train
  train --e=500
  train --t=10m
  train --reduce-percent=20
  train --e=500 --validation-split=0.2 --patience=12 --seed=42
  train --e=100 --checkpoint=checkpoints/model.json
  train --e=100 --resume=checkpoints/model.json
  train --r=0.15 --t=15m
  train --t=10m --b=32 --lr=0.001 -v
""",
        "stats": """
Model statistics
  stats                   Show architecture, parameter count, and file status
  stats -d, --detailed    Also show output classes and latest training metrics

Example
  stats --detailed
""",
        "history": """
Training history
  history           Show the 10 most recent runs
  history <count>   Show the requested number of recent runs

Example
  history 5
""",
        "cd": """
Change model directory
  cd <index>    Enter a model by its number from 'models'
  cd ~          Return to the user directory
  back          Return to the user directory

Example
  models
  cd 3
""",
        "models": """
List models
  models    List configured models and whether weights are saved
  list      Alias for models
""",
        "help": """
Show command help
  help            Show the concise command summary
  help --<topic> Show detailed usage and examples for a command

Example
  help --train
""",
        "create": """
Create a model
  create    Interactively configure and train a new model

The command asks for a unique name, dataset, architecture, and training
settings, then saves the configuration and model when training succeeds.
Its training setup also supports a reproducibility seed, validation split,
early-stopping patience, and optional checkpoint/resume.
""",
        "predict": """
Run a model
  predict    Run the selected model on input data
  use        Alias for predict

Enter a model directory with 'cd <index>' first.
""",
        "back": """
Return to the user directory
  back    Leave the current model directory and return to the user prompt
  cd ~    Alias for back
""",
        "exit": """
Exit the model manager
  exit    Leave the interactive model manager
  quit    Alias for exit
""",
    }
    topic_aliases = {
        "continue": "train",
        "list": "models",
        "use": "predict",
    }
    resolved_topic = topic_aliases.get(topic, topic)
    if resolved_topic in topics:
        print_help_text(topics[resolved_topic])
        return
    if topic is not None:
        supported_topics = ", ".join(sorted(topics))
        print(
            colorize(
                f"No detailed help for '{topic}'. "
                f"Available topics: {supported_topics}.",
                "warning",
            )
        )
        return
    print_help_text(
        """
Commands
  help                 Show this command summary
  help --<topic>       Show detailed usage and examples for a command
  models, list         List configured models and their status
  cd <index>           Enter a model directory from the numbered list
  stats                Show model details and optionally latest run metrics
  history <count>      Show recent training metrics for the current model
  create               Create and train a new model (interactive setup)
  predict, use         Run the selected model on an input
  train                Review saved settings or train with explicit options
                       (validation, reproducible seeds, and checkpoints supported)
  exit, quit           Leave the model manager

"""
    )


def handle_cd(context: ShellContext) -> ModelSelection | None:
    """Change the shell's selected model directory or return to the user level."""
    if len(context.arguments) != 1:
        print(colorize("Usage: cd <index> or cd ~", "error"))
        return context.selected
    if context.arguments[0] == "~":
        return None
    selection = resolve_model(context.configs, context.arguments[0])
    return context.selected if selection is None else selection


def handle_stats(context: ShellContext) -> ModelSelection | None:
    """Display model statistics, optionally including detailed run metrics."""
    if context.selected is None:
        print(colorize("Enter a model directory first with 'cd <index>'.", "warning"))
    elif context.arguments not in ([], ["--detailed"], ["-d"]):
        print(colorize("Usage: stats [-d | --detailed]", "error"))
    else:
        name = context.selected[0]
        config = context.selected[1]
        show_model_details(name, config, detailed=bool(context.arguments))
    return context.selected


def handle_create(context: ShellContext) -> ModelSelection | None:
    """Create a new model and select it when creation succeeds."""
    if context.arguments:
        print(colorize("Usage: create", "error"))
        return context.selected
    created = create_new_model(context.configs)
    return context.selected if created is None else created


def handle_predict(context: ShellContext) -> ModelSelection | None:
    """Run prediction for the selected model, if one is selected."""
    if context.arguments:
        print(colorize(f"Usage: {context.command}", "error"))
    elif context.selected is None:
        print(colorize("Enter a model directory first with 'cd <index>'.", "warning"))
    else:
        name = context.selected[0]
        config = context.selected[1]
        use_model(name, config)
    return context.selected


def estimate_training_limits(options, has_target, target_epochs):
    """Choose a finite epoch cap using explicit settings and target history."""
    if options.epochs is not None:
        return options.epochs
    if not has_target or options.time is not None:
        return MAX_TIME_BUDGET_EPOCHS
    if target_epochs is None:
        return MAX_TARGET_EPOCHS
    estimate_with_buffer = max(
        math.ceil(target_epochs) + 5,
        math.ceil(target_epochs * TARGET_EPOCH_ESTIMATE_MULTIPLIER),
    )
    return min(MAX_TARGET_EPOCHS, estimate_with_buffer)


def print_target_estimate(options, preview):
    """Explain historical target estimates and their epoch safety limit."""
    if not preview.has_target:
        return
    if preview.target_epochs is None:
        if options.epochs is None and options.time is None:
            print(
                colorize(
                    "No comparable run reached this loss target; using a "
                    f"safety limit of {preview.epoch_limit} epochs.",
                    "warning",
                )
            )
        else:
            print(
                colorize(
                    "No comparable run reached this loss target; the supplied "
                    "epoch/time limit will bound this run.",
                    "warning",
                )
            )
        return

    duration = (
        f", about {preview.target_epochs * preview.epoch_time:.1f}s"
        if preview.epoch_time is not None
        else ""
    )
    print(
        colorize(
            f"Historical target estimate: about {preview.target_epochs:g} epochs "
            f"from {preview.target_runs} comparable run(s){duration}.",
            "accent",
        )
    )
    if options.time is None and options.epochs is None:
        print(colorize(f"  Target safety cap: {preview.epoch_limit} epochs.", "muted"))


def print_training_preview(options, preview):
    """Print expected run duration, epoch caps, and requested loss target."""
    if preview.has_target:
        print_target_estimate(options, preview)
    if preview.has_target and options.epochs is not None:
        print(colorize(f"  Epoch cap: {preview.epoch_limit} epochs.", "muted"))
    if preview.has_target and options.time is None:
        print(
            colorize(
                "Training preview: stop at the loss target or after "
                f"{preview.epoch_limit} epochs.",
                "accent",
            )
        )
    if options.time is None and not preview.has_target:
        expected_duration = (
            f"about {preview.epoch_limit * preview.epoch_time:.1f}s"
            if preview.epoch_time is not None
            else "unknown (no comparable run history)"
        )
        print(
            colorize(
                f"Training preview: {preview.epoch_limit} epochs; {expected_duration}.",
                "accent",
            )
        )
    if options.time is not None:
        print_time_budget_preview(options, preview)


def print_time_budget_preview(options, preview):
    """Print the timed-run estimate, with a clear no-history fallback."""
    if preview.epoch_time is None:
        print(
            colorize(
                f"Training preview: time-limited to about "
                f"{format_duration(options.time)}. No comparable run history; "
                "the first epoch provides the initial timing data.",
                "accent",
            )
        )
        return
    print(
        colorize(
            f"Training preview: time-limited to about "
            f"{format_duration(options.time)}; roughly "
            f"{preview.estimated_epochs} epochs fit based on "
            f"{preview.comparable_runs} comparable run(s) "
            f"({preview.epoch_time:.2f}s/epoch). "
            f"Epoch cap: {preview.epoch_limit}.",
            "accent",
        )
    )


def prepare_training_plan(name, config, options, preferences):
    """Validate training files, show estimates, and resolve a training plan."""
    model_path = model_path_for(config)
    data_path = data_path_for(config)
    if not model_path.is_file():
        print(
            colorize(
                f"Model file not found: {model_path}. Use 'create' first.",
                "error",
            )
        )
        return None
    if not data_path.is_file():
        print(colorize(f"Training data file not found: {data_path}", "error"))
        return None
    validate_checkpoint_paths(config, (options.checkpoint, options.resume))

    batch_size = options.batch_size or preferences["batch_size"]
    learning_rate = options.learning_rate or preferences["learning_rate"]
    epoch_time, comparable_runs = get_epoch_time_estimate(name, config, batch_size)
    estimated_epochs = (
        estimate_epochs_for_budget(options.time, epoch_time)
        if options.time is not None
        else None
    )
    target = LossReductionTarget(
        absolute=options.reduce,
        percentage=options.reduce_percent,
    )
    has_target = target.absolute is not None or target.percentage is not None
    target_epochs, target_runs = (None, 0)
    if has_target:
        target_epochs, target_runs = get_target_epoch_estimate(
            name, config, batch_size, learning_rate, target
        )
    epoch_limit = estimate_training_limits(options, has_target, target_epochs)
    preview = TrainingPreview(
        epoch_time=epoch_time,
        comparable_runs=comparable_runs,
        estimated_epochs=estimated_epochs,
        target_epochs=target_epochs,
        target_runs=target_runs,
        epoch_limit=epoch_limit,
        has_target=has_target,
    )
    print_training_preview(options, preview)
    print(
        colorize(
            f"  Model: {name} | Batch size: {batch_size} | "
            f"Learning rate: {learning_rate:g}",
            "accent",
        )
    )
    if target.absolute is not None:
        print(
            colorize(
                f"  Stop target: loss reduction of at least {target.absolute:g}.",
                "accent",
            )
        )
    elif target.percentage is not None:
        print(
            colorize(
                f"  Stop target: loss reduction of at least {target.percentage:g}%.",
                "accent",
            )
        )
    if options.time is not None:
        print(
            colorize(
                "  Time budgets stop at epoch boundaries; final metric "
                "evaluation may extend the run slightly.",
                "muted",
            )
        )
    if options.validation_split:
        print(
            colorize(
                f"  Validation: {options.validation_split:.0%} holdout; "
                f"early-stopping patience {options.patience}.",
                "accent",
            )
        )
    if options.seed is not None:
        print(colorize(f"  Random seed: {options.seed}.", "accent"))
    if options.checkpoint or options.resume:
        checkpoint_path = options.checkpoint or options.resume
        action = "Resume and update" if options.resume else "Save"
        print(colorize(f"  {action} checkpoint: {checkpoint_path}.", "accent"))
    return TrainingPlan(
        epoch_limit=epoch_limit,
        batch_size=batch_size,
        learning_rate=learning_rate,
        time_budget=options.time,
        requested_epochs=options.epochs,
        target_loss_reduction=target.absolute,
        target_loss_reduction_percent=target.percentage,
        verbose=options.verbose,
        validation_split=options.validation_split,
        patience=options.patience,
        seed=options.seed,
        checkpoint_path=options.checkpoint,
        resume_from=options.resume,
    )


def handle_train(context: ShellContext) -> ModelSelection | None:
    """Resolve training settings interactively or from arguments, then train."""
    if context.arguments in (["--help"], ["-h"]):
        print_help("train")
        return context.selected
    if context.selected is None:
        print(colorize("Enter a model directory first with 'cd <index>'.", "warning"))
        return context.selected

    name, config = context.selected
    preferences = load_training_preferences(name)
    options = (
        get_interactive_training_options(name, preferences)
        if not context.arguments
        else parse_training_options(context.arguments)
    )
    if options is None:
        return context.selected
    try:
        plan = prepare_training_plan(name, config, options, preferences)
    except (OSError, ValueError) as error:
        print(colorize(f"Training setup failed: {error}", "error"))
        return context.selected
    if plan is not None and (options.yes or confirm_training()):
        try:
            continue_training(name, config, plan)
        except (OSError, ValueError) as error:
            print(colorize(f"Training failed: {error}", "error"))
    return context.selected


def get_interactive_training_options(name, preferences):
    """Show the model's saved training defaults and let the user adjust them."""
    default_epochs = preferences["epochs"]
    default_arguments = [f"--e={default_epochs}"] if default_epochs is not None else []
    epoch_setting = (
        f"--e={default_epochs}" if default_epochs is not None else "epoch limit not set"
    )
    print(colorize(f"Interactive training setup: {name}", "header", bold=True))
    print(
        colorize(
            f"Suggested settings: {epoch_setting} "
            f"--b={preferences['batch_size']} "
            f"--lr={preferences['learning_rate']:g}",
            "accent",
        )
    )
    if default_epochs is None:
        print(
            colorize(
                "No previous epoch limit is saved. Enter an objective such as "
                "--e=500 or --t=10m.",
                "warning",
            )
        )
    objective_options = {
        "-e",
        "--e",
        "--epochs",
        "-t",
        "--t",
        "--time",
        "-r",
        "--r",
        "--reduce",
        "--reduce-percent",
    }
    while True:
        try:
            change_line = input(
                colorize(
                    "Enter options to change, or press Enter to use saved settings "
                    "(type 'cancel' to stop): ",
                    "prompt",
                )
            )
        except EOFError:
            print(colorize("\nTraining setup cancelled.", "warning"))
            return None
        if change_line.strip().lower() in {"cancel", "q", "quit"}:
            print(colorize("Training setup cancelled.", "warning"))
            return None
        try:
            changes = shlex.split(change_line)
        except ValueError as error:
            print(colorize(f"Could not parse training options: {error}", "error"))
            continue

        replaces_objective = any(
            argument.split("=", maxsplit=1)[0] in objective_options
            for argument in changes
        )
        arguments = changes if replaces_objective else [*default_arguments, *changes]
        if not arguments:
            print(
                colorize(
                    "No saved epoch limit. Enter a training objective such as "
                    "--e=500 or --t=10m.",
                    "error",
                )
            )
            continue
        options = parse_training_options(arguments)
        if options is not None:
            return options


def handle_history(context: ShellContext) -> ModelSelection | None:
    """Display recent persisted training metrics for the selected model."""
    if context.selected is None:
        print(colorize("Enter a model directory first with 'cd <index>'.", "warning"))
        return context.selected
    if len(context.arguments) > 1:
        print(colorize("Usage: history [count]", "error"))
        return context.selected
    try:
        count = positive_int(context.arguments[0]) if context.arguments else 10
    except argparse.ArgumentTypeError as error:
        print(
            colorize(f"Invalid history count: {error}\nUsage: history [count]", "error")
        )
        return context.selected

    name = context.selected[0]
    runs = load_training_history().get(name, [])[-count:]
    if not runs:
        print(colorize(f"No training history for '{name}' yet.", "warning"))
        return context.selected
    print(colorize(f"\nRecent training runs for {name}", "header", bold=True))
    for run in runs:
        duration = format_duration(run["duration_seconds"])
        seconds_per_epoch = run.get("seconds_per_epoch")
        pace = (
            f"{seconds_per_epoch:.2f}s/epoch"
            if seconds_per_epoch is not None
            else "n/a"
        )
        reduction = run.get("loss_reduction_percent")
        reduction_text = f"{reduction:+.2f}%" if reduction is not None else "n/a"
        print(
            colorize(
                f"  {run['started_at']} | {run['epochs_completed']} epochs | "
                f"batch {run['batch_size']} | lr {run['learning_rate']:g} | "
                f"{duration} ({pace})",
                "accent",
            )
        )
        print(
            colorize(
                f"    loss {run['initial_loss']:.6f} -> {run['final_loss']:.6f} "
                f"({run['loss_reduction']:+.6f}, {reduction_text}); "
                f"training accuracy {run['training_accuracy']:.1%}",
                "accent",
            )
        )
        print_history_training_extras(run)
        target_reduction = run.get("target_loss_reduction")
        target_percent = run.get("target_loss_reduction_percent")
        if target_reduction is not None:
            target_text = f"loss reduction >= {target_reduction:g}"
        elif target_percent is not None:
            target_text = f"loss reduction >= {target_percent:g}%"
        else:
            target_text = None
        if target_text is not None:
            print(
                colorize(
                    f"    target: {target_text}; reached: "
                    f"{'yes' if run.get('target_reached') else 'no'}",
                    "success" if run.get("target_reached") else "warning",
                )
            )
        if run.get("time_budget_seconds") is not None:
            print(
                colorize(
                    f"    time budget {format_duration(run['time_budget_seconds'])}; "
                    f"budget reached: {'yes' if run['time_budget_reached'] else 'no'}",
                    "warning" if run["time_budget_reached"] else "muted",
                )
            )
    return context.selected


def print_history_training_extras(run):
    """Display validation metrics and reproducibility details for one run."""
    validation_accuracy = run.get("validation_accuracy")
    validation_loss = run.get("validation_loss")
    if validation_accuracy is not None and validation_loss is not None:
        print(
            colorize(
                f"    validation loss {validation_loss:.6f}; "
                f"accuracy {validation_accuracy:.1%}; "
                f"best epoch {run.get('best_validation_epoch', 'n/a')}; "
                f"split {run.get('validation_split', 0):.0%}",
                "accent",
            )
        )
    if run.get("seed") is not None:
        print(colorize(f"    seed {run['seed']}", "muted"))
    if run.get("stopped_by_early_stopping"):
        print(
            colorize(
                "    stopped early; restored best validation weights",
                "success",
            )
        )


def handle_back(context: ShellContext) -> ModelSelection | None:
    """Return to the user-level prompt."""
    if context.arguments:
        print(colorize("Usage: back", "error"))
        return context.selected
    return None


def handle_help(context: ShellContext) -> ModelSelection | None:
    """Display help without changing the current selection."""
    if not context.arguments:
        print_help()
    elif len(context.arguments) == 1 and context.arguments[0].startswith("--"):
        topic = context.arguments[0][2:].lower()
        if topic:
            print_help(topic)
        else:
            print(colorize("Usage: help [--<topic>]", "error"))
    else:
        print(colorize("Usage: help [--<topic>]", "error"))
    return context.selected


def handle_models(context: ShellContext) -> ModelSelection | None:
    """List all models without changing the current selection."""
    print_model_list(context.configs)
    return context.selected


def run_shell(configs: ModelConfigs) -> None:
    """Run the interactive command loop until the user exits."""
    selected: ModelSelection | None = None
    handlers: dict[str, CommandHandler] = {
        "help": handle_help,
        "models": handle_models,
        "list": handle_models,
        "cd": handle_cd,
        "stats": handle_stats,
        "history": handle_history,
        "create": handle_create,
        "predict": handle_predict,
        "use": handle_predict,
        "train": handle_train,
        "continue": handle_train,
        "back": handle_back,
    }
    while True:
        prompt = "user/" if selected is None else f"user/{selected[0]}/"
        try:
            command_line = input(
                f"{colorize(prompt, 'prompt', bold=True)} "
                f"{colorize('>', 'accent', bold=True)} "
            ).strip()
        except EOFError:
            print(colorize("\nExiting model manager.", "muted"))
            return
        if not command_line:
            continue
        try:
            parts = shlex.split(command_line)
        except ValueError as error:
            print(colorize(f"Could not parse command: {error}", "error"))
            continue

        command, *arguments = parts
        command = command.lower()
        if command in {"exit", "quit"}:
            print(colorize("Exiting model manager.", "muted"))
            return
        handler = handlers.get(command)
        if handler is None:
            print(
                colorize(
                    f"Unknown command: {command}. "
                    "Enter 'help' to see available commands.",
                    "error",
                )
            )
            continue
        context = ShellContext(command, arguments, configs, selected)
        selected = handler(context)


def main():
    """Load model configurations and start the interactive shell."""
    print(colorize("\nNeural Network Manager \n", "header", bold=True))
    configs = load_configs()
    # print_model_list(configs)
    # print("\n")
    print(colorize("Enter 'help' for available commands.", "muted"))
    run_shell(configs)
