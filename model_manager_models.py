"""Model configuration display, creation, and prediction workflows."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

import tic_tac_toe
from model_manager_training import (
    DATA_DIR,
    ModelConfigs,
    ModelSelection,
    TrainingJob,
    TrainingPlan,
    ask_float,
    ask_int,
    confirm_training,
    data_path_for,
    dataset_hash_for,
    format_duration,
    load_data,
    load_training_history,
    model_path_for,
    non_negative_int,
    positive_int,
    resolve_asset_path,
    save_configs,
    train_and_record,
    validate_checkpoint_paths,
    validate_training_data,
    validation_fraction,
)
from neural_network import NeuralNetwork, argmax, softmax
from terminal_colors import colorize


def print_model_list(configs):
    """Display configured models with their current on-disk status."""
    print(colorize("\nAvailable models", "header", bold=True))
    print(colorize("----------------", "muted"))
    if not configs:
        print("No models are configured. Use 'create' to add one.")
        return
    for index, (name, config) in enumerate(configs.items(), 1):
        model_path = model_path_for(config)
        status = (
            colorize("ready", "success")
            if model_path.is_file()
            else colorize("not trained", "warning")
        )
        print(f"  {index:>2}. {name}  [{status}]")


def show_model_details(name, config, detailed=False):
    """Display model configuration and, in detailed mode, classes and run metrics."""
    model_path = model_path_for(config)
    data_path = data_path_for(config)
    hidden_sizes = config.get("hidden_sizes", [])
    layer_sizes = [config["input_size"], *hidden_sizes, config["output_size"]]
    architecture = " -> ".join(str(size) for size in layer_sizes)
    parameter_count = sum(
        (input_size + 1) * output_size
        for input_size, output_size in zip(layer_sizes, layer_sizes[1:], strict=False)
    )
    input_mode = {
        "binary": "Binary number",
        "tic_tac_toe": "Tic-tac-toe board",
    }.get(config.get("input_mode"), "Numeric")
    rows = [
        ("Architecture", architecture),
        ("Trainable parameters", f"{parameter_count:,}"),
        ("Input format", input_mode),
        ("Input scale", str(config.get("input_scale", 1))),
        (
            "Model file",
            f"{config['model_file']} "
            f"({'ready' if model_path.is_file() else 'not trained'})",
        ),
        (
            "Training data",
            f"{config['data_file']} "
            f"({'available' if data_path.is_file() else 'missing'})",
        ),
    ]
    label_width = max(len(label) for label, _ in rows)
    content_rows = [f"  {label:<{label_width}}  {value}" for label, value in rows]
    classes = config.get("class_names", [])
    if detailed:
        content_rows.append("")
        content_rows.append("  Output classes")
        if classes:
            content_rows.extend(
                f"  {index:>2}. {class_name}"
                for index, class_name in enumerate(classes)
            )
        else:
            content_rows.append("  No output classes are configured.")
        content_rows.append("")
        content_rows.append("  Latest training run")
        runs = load_training_history().get(name, [])
        if runs:
            latest_run = runs[-1]
            duration = latest_run.get("duration_seconds")
            seconds_per_epoch = latest_run.get("seconds_per_epoch")
            loss_reduction = latest_run.get("loss_reduction")
            loss_reduction_percent = latest_run.get("loss_reduction_percent")
            run_rows = [
                ("Started", latest_run.get("started_at", "Unknown")),
                ("Epochs", str(latest_run.get("epochs_completed", "Unknown"))),
                (
                    "Duration",
                    f"{duration:.2f}s" if duration is not None else "Unknown",
                ),
                (
                    "Time per epoch",
                    f"{seconds_per_epoch:.2f}s"
                    if seconds_per_epoch is not None
                    else "Unknown",
                ),
                ("Dataset size", str(latest_run.get("dataset_size", "Unknown"))),
                ("Batch size", str(latest_run.get("batch_size", "Unknown"))),
                ("Learning rate", f"{latest_run['learning_rate']:g}")
                if isinstance(latest_run.get("learning_rate"), (int, float))
                else ("Learning rate", "Unknown"),
                (
                    "Loss",
                    (
                        f"{latest_run['initial_loss']:.6f} -> "
                        f"{latest_run['final_loss']:.6f}"
                    )
                    if isinstance(latest_run.get("initial_loss"), (int, float))
                    and isinstance(latest_run.get("final_loss"), (int, float))
                    else "Unknown",
                ),
                (
                    "Loss reduction",
                    (
                        f"{loss_reduction:.6f} ({loss_reduction_percent:.2f}%)"
                        if loss_reduction_percent is not None
                        else f"{loss_reduction:.6f}"
                    )
                    if loss_reduction is not None
                    else "Unknown",
                ),
                (
                    "Training accuracy",
                    f"{latest_run['training_accuracy']:.1%}"
                    if isinstance(latest_run.get("training_accuracy"), (int, float))
                    else "Unknown",
                ),
            ]
            early_stopped = append_reproducibility_metrics(run_rows, latest_run)
            content_rows.extend(f"  {label:<20}  {value}" for label, value in run_rows)
            if early_stopped:
                content_rows.append(
                    "  Early stopping       Best validation weights restored"
                )
            if latest_run.get("target_loss_reduction") is not None:
                content_rows.append(
                    "  Stop target          "
                    f"Loss reduction >= {latest_run['target_loss_reduction']:g}; "
                    f"reached: {'yes' if latest_run.get('target_reached') else 'no'}"
                )
            elif latest_run.get("target_loss_reduction_percent") is not None:
                content_rows.append(
                    "  Stop target          "
                    f"Loss reduction >= "
                    f"{latest_run['target_loss_reduction_percent']:g}%; "
                    f"reached: {'yes' if latest_run.get('target_reached') else 'no'}"
                )
            if latest_run.get("time_budget_seconds") is not None:
                content_rows.append(
                    "  Time budget          "
                    f"{format_duration(latest_run['time_budget_seconds'])}; "
                    "reached: "
                    f"{'yes' if latest_run.get('time_budget_reached') else 'no'}"
                )
        else:
            content_rows.append("  No training history is recorded for this model.")

    print_model_details_box(f"Model: {name}", content_rows)


def print_model_details_box(title, content_rows):
    """Render aligned model information and highlight key metric sections."""
    content_width = max(len(title), *(len(row) for row in content_rows))
    print(colorize(f"\n+{'-' * (content_width + 2)}+", "muted"))
    print(colorize(f"| {title:<{content_width}} |", "header", bold=True))
    for row in content_rows:
        formatted_row = f"| {row:<{content_width}} |"
        if row.lstrip().startswith(("Output classes", "Latest training run")):
            formatted_row = colorize(formatted_row, "accent", bold=True)
        elif row.lstrip().startswith(("Training accuracy", "Loss reduction")):
            formatted_row = colorize(formatted_row, "success")
        print(formatted_row)
    print(colorize(f"+{'-' * (content_width + 2)}+", "muted"))


def append_reproducibility_metrics(run_rows, latest_run):
    """Add validation and reproducibility data to the detailed run display."""
    validation_loss = latest_run.get("validation_loss")
    validation_accuracy = latest_run.get("validation_accuracy")
    if validation_loss is not None or validation_accuracy is not None:
        run_rows.extend(
            [
                ("Validation split", f"{latest_run.get('validation_split', 0):.0%}"),
                (
                    "Training examples",
                    str(latest_run.get("training_dataset_size", "Unknown")),
                ),
                (
                    "Validation examples",
                    str(latest_run.get("validation_dataset_size", "Unknown")),
                ),
                (
                    "Validation loss",
                    (
                        f"{validation_loss:.6f}"
                        if validation_loss is not None
                        else "Unknown"
                    ),
                ),
                (
                    "Validation accuracy",
                    f"{validation_accuracy:.1%}"
                    if validation_accuracy is not None
                    else "Unknown",
                ),
                (
                    "Best validation epoch",
                    str(latest_run.get("best_validation_epoch", "Unknown")),
                ),
            ]
        )
    run_rows.extend(
        [
            ("Random seed", str(latest_run.get("seed", "None"))),
            (
                "Dataset SHA-256",
                (
                    f"{latest_run['dataset_sha256'][:16]}..."
                    if latest_run.get("dataset_sha256")
                    else "Unknown"
                ),
            ),
            ("Checkpoint", str(latest_run.get("checkpoint_file") or "None")),
        ]
    )
    return latest_run.get("stopped_by_early_stopping", False)


def resolve_model(configs: ModelConfigs, argument: str) -> ModelSelection | None:
    """Resolve a one-based model index to its name and configuration."""
    if argument.isdigit():
        index = int(argument)
    elif argument.startswith("-") and argument[1:].isdigit():
        index = int(argument[1:])
    else:
        print(
            colorize(
                "Model index must be a positive integer, for example: cd 3", "error"
            )
        )
        return None

    entries = list(configs.items())
    if not 1 <= index <= len(entries):
        if entries:
            print(
                colorize(f"Model index must be between 1 and {len(entries)}.", "error")
            )
        else:
            print(colorize("No models are configured.", "warning"))
        return None
    return entries[index - 1]


def prompt_training_extras():
    """Prompt for optional reproducibility and validation settings."""
    while True:
        seed_text = input(
            colorize("Random seed (blank for an unseeded run): ", "prompt")
        ).strip()
        try:
            seed = non_negative_int(seed_text) if seed_text else None
            break
        except argparse.ArgumentTypeError as error:
            print(colorize(f"Invalid seed: {error}", "error"))
    while True:
        split_text = input(
            colorize("Validation split (0 disables validation) [0]: ", "prompt")
        ).strip()
        try:
            validation_split = validation_fraction(split_text or "0")
            break
        except argparse.ArgumentTypeError as error:
            print(colorize(f"Invalid validation split: {error}", "error"))
    patience = 10
    if validation_split:
        while True:
            patience_text = input(
                colorize("Early-stopping patience [10]: ", "prompt")
            ).strip()
            try:
                patience = positive_int(patience_text or "10")
                break
            except argparse.ArgumentTypeError as error:
                print(colorize(f"Invalid patience: {error}", "error"))
    return seed, validation_split, patience


def prompt_checkpoint_options(config):
    """Offer opt-in checkpointing and explicit resume of existing files."""
    while True:
        filename = input(
            colorize("Checkpoint file (blank disables checkpointing): ", "prompt")
        ).strip()
        if not filename:
            return None, None
        try:
            validate_checkpoint_paths(config, (filename,))
        except ValueError as error:
            print(colorize(str(error), "error"))
            continue
        path = Path(filename).expanduser()
        if path.exists():
            if not path.is_file():
                print(colorize("Checkpoint path must be a file.", "error"))
                continue
            answer = (
                input(
                    colorize(
                        "Checkpoint exists. Resume from it? [y/N] ",
                        "warning",
                    )
                )
                .strip()
                .lower()
            )
            if answer in {"y", "yes"}:
                return filename, filename
            print(
                colorize("Choose a different checkpoint path or press Enter.", "muted")
            )
            continue
        return filename, None


def dataset_dimensions(raw_data, data_path):
    """Validate a new dataset and infer its input and output sizes."""
    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError(f"Training data file is empty or invalid: {data_path}")
    first = raw_data[0]
    if (
        not isinstance(first, dict)
        or not isinstance(first.get("inputs"), list)
        or not isinstance(first.get("targets"), list)
    ):
        raise ValueError(
            f"Training data must contain examples with input and target lists: "
            f"{data_path}"
        )
    input_size = len(first["inputs"])
    output_size = len(first["targets"])
    validate_training_data(
        raw_data,
        {"input_size": input_size, "output_size": output_size},
        data_path,
    )
    return input_size, output_size


def prompt_creation_dataset():
    """Select and preflight a dataset for a new model."""
    data_file = input(colorize("Training data file: ", "prompt")).strip()
    data_path = resolve_asset_path(data_file, DATA_DIR)
    if not data_path.exists():
        print(colorize(f"Training data file not found: {data_path}", "error"))
        return None
    try:
        with open(data_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except (OSError, ValueError) as error:
        print(colorize(f"Could not read training data: {error}", "error"))
        return None
    try:
        input_size, output_size = dataset_dimensions(raw_data, data_path)
    except ValueError as error:
        print(colorize(f"Invalid training data: {error}", "error"))
        return None
    print(
        colorize(f"Detected {input_size} inputs and {output_size} outputs.", "accent")
    )
    return data_file, input_size, output_size


def prompt_model_architecture(data_file, input_size, output_size):
    """Collect architecture, scaling, and class labels for a new model."""
    hidden_count = ask_int("Number of hidden layers: ", 0)
    hidden_sizes = [
        ask_int(f"Neurons in hidden layer {index + 1}: ")
        for index in range(hidden_count)
    ]
    input_scale = ask_float("Input scale (1 means no scaling): ", 0.000001)
    model_file = input(colorize("Model file name: ", "prompt")).strip()
    while not model_file:
        model_file = input(colorize("Model file name: ", "prompt")).strip()

    class_names = []
    for index in range(output_size):
        class_name = input(colorize(f"Name for class {index}: ", "prompt")).strip()
        class_names.append(class_name or f"Class {index}")

    return {
        "data_file": data_file,
        "model_file": model_file,
        "input_size": input_size,
        "output_size": output_size,
        "hidden_sizes": hidden_sizes,
        "input_scale": input_scale,
        "class_names": class_names,
    }


def prepare_creation_job(name, config, data, dataset_hash):
    """Collect initial training settings and return a confirmed model job."""
    epochs = ask_int("Training epochs: ")
    batch_size = ask_int("Batch size: ")
    learning_rate = ask_float("Learning rate: ", 0.000001)
    seed, validation_split, patience = prompt_training_extras()
    checkpoint_path, resume_from = prompt_checkpoint_options(config)
    network = NeuralNetwork(
        config["input_size"], config["hidden_sizes"], config["output_size"], seed=seed
    )
    print(
        colorize(
            f"Training preview: {epochs} epochs, {len(data)} examples, "
            f"batch size {batch_size}, learning rate {learning_rate:g}.",
            "accent",
        )
    )
    if validation_split:
        print(
            colorize(
                f"Validation holdout: {validation_split:.0%}; "
                f"early-stopping patience: {patience} epochs.",
                "accent",
            )
        )
    if seed is not None:
        print(colorize(f"Reproducible random seed: {seed}.", "accent"))
    if checkpoint_path:
        action = "Resume and update" if resume_from else "Save"
        print(colorize(f"{action} checkpoint: {checkpoint_path}.", "accent"))
    if not confirm_training():
        print(colorize("Model creation cancelled.", "warning"))
        return None
    plan = TrainingPlan(
        epoch_limit=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        requested_epochs=epochs,
        validation_split=validation_split,
        patience=patience,
        seed=seed,
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
    )
    return TrainingJob(
        name=name,
        config=config,
        network=network,
        training_data=data,
        plan=plan,
        dataset_hash=dataset_hash,
    )


def create_new_model(configs):
    """Interactively configure, train, and save a new model."""
    name = input(colorize("New model name: ", "prompt")).strip()
    while not name or name in configs:
        print(colorize("The name is empty or already in use.", "error"))
        name = input(colorize("New model name: ", "prompt")).strip()

    dataset = prompt_creation_dataset()
    if dataset is None:
        return None
    data_file, input_size, output_size = dataset
    config = prompt_model_architecture(data_file, input_size, output_size)
    try:
        data = load_data(config)
        dataset_hash = dataset_hash_for(config)
    except (OSError, ValueError) as error:
        print(colorize(f"Could not prepare training data: {error}", "error"))
        return None
    job = prepare_creation_job(name, config, data, dataset_hash)
    if job is None:
        return None
    try:
        train_and_record(job)
    except (OSError, ValueError) as error:
        print(colorize(f"Training failed: {error}", "error"))
        return None
    configs[name] = config
    save_configs(configs)
    print(colorize(f"Model '{name}' created and saved.", "success"))
    return name, config


def predict(network, config):
    """Collect model-specific input and print the predicted class probabilities."""
    if config.get("input_mode") == "tic_tac_toe":
        play_tic_tac_toe(network)
        return
    if config.get("input_mode") == "binary":
        number = ask_int("Number to classify: ", 1)
        values = [
            int(bit)
            for bit in f"{number:0{config['input_size']}b}"[-config["input_size"] :]
        ]
    else:
        values = [
            ask_float(f"Input {index + 1}: ") for index in range(config["input_size"])
        ]
    input_scale = config.get("input_scale", 1)
    probabilities = softmax(network.forward([value / input_scale for value in values]))
    prediction = argmax(probabilities)
    labels = config.get("class_names", [])
    label = labels[prediction] if prediction < len(labels) else f"Class {prediction}"
    print(colorize(f"Prediction: {label}", "success", bold=True))
    print(
        colorize(
            "Probabilities: "
            + ", ".join(
                f"{labels[index] if index < len(labels) else f'Class {index}'}: "
                f"{probability:.2%}"
                for index, probability in enumerate(probabilities)
            ),
            "accent",
        )
    )


def play_tic_tac_toe(network):
    """Play a game against the supplied Tic-Tac-Toe neural network."""
    game_board = np.zeros((3, 3), dtype=int)
    winner = ""
    ai_starts = random.choice((True, False))
    human_player = 2 if ai_starts else 1
    ai_player = 1 if ai_starts else 2
    current_player = ai_player if ai_starts else human_player
    print(
        colorize(
            f"You are {'O' if ai_starts else 'X'}. "
            "Enter the row and column, each from 0 to 2.",
            "accent",
        )
    )

    while True:
        if not (ai_starts and np.count_nonzero(game_board) == 0):
            tic_tac_toe.draw_game_board(game_board)
            print()

        if current_player == human_player:
            move_possible = False
            row = 0
            column = 0
            while not move_possible:
                move_possible, row, column = tic_tac_toe.is_move_possible(
                    input(colorize("Your move (row column): ", "prompt")), game_board
                )
                if not move_possible:
                    print(
                        colorize(
                            "Invalid move. Enter the row and column, in that order, "
                            "using values from 0 to 2.",
                            "warning",
                        )
                    )
            tic_tac_toe.handle_inputs(row, column, game_board, human_player)
        else:
            encoded_board = tic_tac_toe.encode_board(game_board, ai_player)
            predictions = softmax(network.forward(encoded_board))
            legal_moves = tic_tac_toe.get_legal_moves(game_board)
            if not legal_moves:
                break
            move_index = max(legal_moves, key=lambda index: predictions[index])
            tic_tac_toe.handle_inputs(
                move_index // 3, move_index % 3, game_board, ai_player
            )

        someone_has_won, winner = tic_tac_toe.declare_winner(game_board)
        if someone_has_won:
            break
        current_player = human_player if current_player == ai_player else ai_player

    tic_tac_toe.draw_game_board(game_board)
    print()
    if winner == str(human_player):
        print(colorize("You won!", "success", bold=True))
    elif winner == str(ai_player):
        print(colorize("The ai won.", "error", bold=True))
    else:
        print(colorize("Draw.", "accent", bold=True))


def use_model(name, config):
    """Load an existing model and run its prediction flow."""
    model_path = model_path_for(config)
    if not model_path.exists():
        print(
            colorize(
                f"Model file not found: {model_path}. Create or train the model first.",
                "error",
            )
        )
        return
    print(colorize(f"Loaded model '{name}'.", "success"))
    predict(NeuralNetwork.load(model_path), config)


def continue_training(
    name,
    config,
    plan,
):
    """Continue training an existing model and save the updated weights."""
    model_path = model_path_for(config)
    if not model_path.exists():
        print(
            colorize(
                f"Model file not found: {model_path}. Use 'create' first.", "error"
            )
        )
        return
    data_path = data_path_for(config)
    if not data_path.is_file():
        print(colorize(f"Training data file not found: {data_path}", "error"))
        return
    try:
        data = load_data(config)
        dataset_hash = dataset_hash_for(config)
        network = NeuralNetwork.load(model_path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(colorize(f"Could not prepare training: {error}", "error"))
        return
    run = train_and_record(
        TrainingJob(
            name=name,
            config=config,
            network=network,
            training_data=data,
            plan=plan,
            dataset_hash=dataset_hash,
        )
    )
    print_run_summary(name, run)


def print_run_summary(name, run):
    """Display key outcomes from a completed training run."""
    print(
        colorize(
            f"Model '{name}' trained for {run['epochs_completed']} epochs "
            f"in {run['duration_seconds']:.2f}s.",
            "success",
            bold=True,
        )
    )
    print(
        colorize(
            f"Loss: {run['initial_loss']:.6f} -> {run['final_loss']:.6f} "
            f"({run['loss_reduction_percent'] or 0:.2f}% reduction); "
            f"training accuracy: {run['training_accuracy']:.1%}.",
            "accent",
        )
    )
    if run["time_budget_reached"]:
        print(
            colorize(
                "Stopped at an epoch boundary after reaching the time budget.",
                "warning",
            )
        )
    if run["target_reached"]:
        print(colorize("The requested loss-reduction target was reached.", "success"))
    elif (
        run.get("target_loss_reduction") is not None
        or run.get("target_loss_reduction_percent") is not None
    ):
        if run["epoch_limit_reached"]:
            print(
                colorize(
                    "The epoch limit was reached before meeting the loss target.",
                    "warning",
                )
            )
        elif run["time_budget_reached"]:
            print(
                colorize(
                    "The time budget expired before meeting the loss target.",
                    "warning",
                )
            )
    if run.get("validation_accuracy") is not None:
        print(
            colorize(
                f"Validation loss: {run['validation_loss']:.6f}; "
                f"accuracy: {run['validation_accuracy']:.1%}; "
                f"best epoch: {run['best_validation_epoch']}.",
                "accent",
            )
        )
    if run.get("stopped_by_early_stopping"):
        print(
            colorize(
                "Stopped after validation loss plateaued; restored the best "
                "validation weights.",
                "warning",
            )
        )
