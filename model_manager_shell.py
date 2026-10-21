"""Interactive command handlers and shell loop for the model manager."""

import argparse
import math
import shlex
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime

import connect_four
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
    get_evolutionary_time_estimate,
    get_target_epoch_estimate,
    load_configs,
    load_training_history,
    load_training_preferences,
    model_path_for,
    parse_training_options,
    positive_int,
    save_training_run,
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
  train -t <time>, --t=<time>       Train up to a time budget (e.g. 1"--evolutionary"0m, 1h)
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
  --evolutionary                    Train the model (if supported) using evolutionary algorithms.

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
For models configured with training_method="evolutionary",
the command asks for the number of generations and confirmation.
Standard training options do not apply to these models.

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

For Tic-Tac-Toe and Connect Four, this starts an
interactive game against the selected model.

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
  back, cd ~           Return to the user prompt
  history <count>      Show recent training runs for the selected model

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

    # Evolutionary training for models that use this training method.
    if config.get("training_method") == "evolutionary":
        if context.arguments not in ([], ["--evolutionary"]):
            print(colorize("Usage: train [--evolutionary]", "error"))
            return context.selected

        while True:
            try:
                raw_generations = input("Number of generations: ").strip()
            except EOFError:
                print(colorize("\nTraining cancelled.", "warning"))
                return context.selected

            try:
                generations = positive_int(raw_generations)
                break
            except argparse.ArgumentTypeError as error:
                print(colorize(f"Invalid generation count: {error}", "error"))

        if confirm_training():
            architecture = "-".join(
                str(size)
                for size in (
                    config["input_size"],
                    *config["hidden_sizes"],
                    config["output_size"],
                )
            )
            historical_seconds, historical_runs = get_evolutionary_time_estimate(
                name, architecture
            )

            if historical_seconds is not None:
                print(
                    colorize(
                        f"Historical estimate: "
                        f"{format_duration(historical_seconds * generations)} "
                        f"from {historical_runs} comparable run(s).",
                        "accent",
                    )
                )
            else:
                print(
                    colorize(
                        "No historical Connect Four runtime yet; "
                        "the ETA will adapt to this run.",
                        "muted",
                    )
                )

            started_at = datetime.now().astimezone().isoformat(timespec="seconds")
            start_time = time.monotonic()

            try:
                connect_four.evolutionary_training(
                    generations,
                    seconds_per_generation=historical_seconds,
                )
                duration = time.monotonic() - start_time

                save_training_run(
                    name,
                    {
                        "started_at": started_at,
                        "training_method": "evolutionary",
                        "architecture": architecture,
                        "generations_requested": generations,
                        "duration_seconds": duration,
                        "seconds_per_generation": duration / generations,
                    },
                )
                print("Training runtime saved to history.")
            except (OSError, ValueError, RuntimeError) as error:
                print(colorize(f"Evolutionary training failed: {error}", "error"))

        return context.selected

    # Existing training flow for all other models.
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
            colorize(
                f"Invalid history count: {error}\nUsage: history [count]",
                "error",
            )
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

        if run.get("training_method") == "evolutionary":
            seconds_per_generation = run.get("seconds_per_generation")
            pace = (
                f"{seconds_per_generation:.2f}s/evaluation"
                if seconds_per_generation is not None
                else "n/a"
            )

            print(
                colorize(
                    f"  {run.get('started_at', 'n/a')} | "
                    f"{run.get('generations_requested', '?')} generations | "
                    f"{duration} ({pace})",
                    "accent",
                )
            )
            print(
                colorize(
                    f"    method evolutionary | "
                    f"architecture {run.get('architecture', 'n/a')}",
                    "muted",
                )
            )
            continue

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
