# Neural Network Model Manager

A NumPy-based neural network project built from scratch in Python. It combines a configurable feedforward neural network with an interactive command-line model manager, training utilities, dataset generators, and game-playing experiments.

The project explores neural network implementation and training through practical classification tasks, Tic-Tac-Toe, and an evolutionary Connect Four agent.

## Features

- **Neural network from scratch:** Configurable fully connected networks with hidden layers, weights, biases, forward propagation, and training.
- **Interactive model manager:** Create, inspect, train, and use configured models through a shell-like command interface.
- **Flexible training controls:** Epoch and time limits, loss-reduction targets, batch size, learning rate, and verbose progress reporting.
- **Validation and early stopping:** Optional holdout validation, configurable patience, and restoration of the best validation weights.
- **Reproducible experiments:** Optional random seeds and SHA-256 dataset fingerprints in recorded training history.
- **Checkpointing:** Save versioned training checkpoints and resume compatible runs.
- **Game-playing experiments:** Tic-Tac-Toe and Connect Four implementations, including evolutionary training for Connect Four.
- **Dataset generation:** Scripts for circle classification, prime classification, odd/even classification, and Tic-Tac-Toe training data.
- **Automated tests:** A test suite based on Python's standard-library `unittest` framework.

## Requirements

- Python 3.10 or newer
- NumPy 1.24 or newer

The dependencies are listed in `requirements.txt`. No separate machine-learning framework is required.

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/Matthis2011/Neural-Network.git
cd Neural-Network
```

Create and activate a virtual environment (recommended).

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Launch the model manager from the repository root:

```bash
python main.py
```

The application opens an interactive prompt. Start by listing the configured models:

```text
user/ > models
```

The manager displays each model's name, index, and saved-model status. Select a model using its displayed index:

```text
user/ > cd <index>
```

Replace `<index>` with the number shown by `models`. The prompt then identifies the selected model.

Useful first commands:

```text
user/ > help
user/ > models
user/ > cd <index>
user/<model>/ > stats
user/<model>/ > stats --detailed
user/<model>/ > history
user/<model>/ > help --train
```

Use `back` or `cd ~` to return to the top-level prompt. Use `exit` or `quit` to leave the application.

## Command reference

Commands are entered at the model manager's prompt, not directly in the operating-system terminal.

| Command | Purpose |
|---|---|
| `help` | Show the concise command summary |
| `help --<topic>` | Show detailed help for a supported topic |
| `models`, `list` | List configured models and their saved-model status |
| `cd <index>` | Select a model using its displayed index |
| `cd ~`, `back` | Return to the top-level prompt |
| `stats` | Show model architecture, parameter count, and file status |
| `stats --detailed` | Include output classes and latest training metrics |
| `history [count]` | Display recent recorded training runs; defaults to 10 |
| `create` | Interactively configure, train, and save a new model |
| `train` | Configure or start training for the selected model |
| `continue` | Alias for `train` |
| `predict`, `use` | Run the selected model's prediction workflow |
| `exit`, `quit` | Exit the model manager |

The detailed help topics include `train`, `stats`, `history`, `cd`, `models`, `create`, `predict`, `back`, and `exit`. For example:

```text
user/ > help --train
user/ > help --predict
```

Some commands require a selected model. If no model is selected, first run `models` and then `cd <index>`.

## Training models

### Standard training

Select a model and run:

```text
user/ > models
user/ > cd <index>
user/<model>/ > train
```

Running `train` without arguments opens an interactive setup that shows the saved training defaults. Enter options to adjust them, or press Enter to accept the suggested settings. If no epoch limit has been saved, specify an objective such as an epoch count or time budget.

Training with explicit options uses the same preview and confirmation flow:

```text
train --e=500
train --t=10m
train --e=500 --validation-split=0.2 --patience=12 --seed=42
```

Training normally requires confirmation before it begins. The `-y` or `--yes` option skips this confirmation for supported standard-training commands.

### Training options

| Option | Description |
|---|---|
| `-e`, `--e`, `--epochs` | Set the maximum number of epochs |
| `-t`, `--t`, `--time` | Set a training time budget, for example `10m` or `1h` |
| `-r`, `--r`, `--reduce` | Stop when the requested absolute loss reduction is reached |
| `--reduce-percent` | Stop when the requested percentage loss reduction is reached |
| `-b`, `--b` | Set the batch size |
| `-lr`, `--lr` | Set the learning rate |
| `-y`, `--yes` | Skip the confirmation prompt |
| `-v`, `--verbose` | Print loss for every epoch |
| `--validation-split` | Reserve a fraction of the data for validation |
| `--patience` | Set the number of unimproved validation epochs allowed |
| `--seed` | Set the random seed for supported reproducibility behavior |
| `--checkpoint` | Save a resumable checkpoint after each epoch |
| `--resume` | Resume training from a compatible checkpoint |

Short options and long options are supported. For example, `train -e 100` and `train --epochs=100` specify the same epoch limit.

Loss-reduction targets can be combined with epoch and/or time limits. The limits stop training if reached before the requested loss target.

### Validation and early stopping

Validation is optional and disabled by default. To reserve 20% of the data for validation:

```text
train --e=500 --validation-split=0.2 --patience=10
```

The validation split is used to monitor generalization during training. Early stopping halts a run after the configured number of epochs without improvement in validation loss. When applicable, the model restores the weights from the best validation-loss epoch.

Validation metrics and the best validation epoch are available through `history` and `stats --detailed`.

### Checkpointing and resuming

To save a checkpoint during training:

```text
train --e=100 --checkpoint=checkpoints/model.json
```

To resume a compatible checkpoint:

```text
train --e=100 --resume=checkpoints/model.json
```

The epoch count supplied when resuming represents additional epochs. A resumed run updates its checkpoint in place unless a different checkpoint path is supplied.

Resume compatibility requires the same dataset, batch size, learning rate, validation split, patience, seed, and loss-reduction target. Epoch and time budgets may be changed. Checkpointing is optional.

### Training history

The manager records standard training-run metrics, including duration, epochs completed, batch size, learning rate, loss reduction, and training accuracy. Supported runs also record validation results, random seeds, checkpoint information, and the dataset's SHA-256 digest.

Inspect recent runs with:

```text
history
history 5
```

For a more detailed view of the latest run, use:

```text
stats --detailed
```

Training accuracy and decreasing loss are useful diagnostics, but neither alone proves that a model generalizes well or performs strongly in a game.

## Connect Four

Connect Four is configured as an evolutionary-training model in `config/model_configs.json`. Its current architecture has 42 inputs, two hidden layers of 64 neurons each, and 7 outputs—one output for each column.

### Start evolutionary training

1. Launch the model manager with `python main.py`.
2. Run `models` and identify the entry named `Connect-Four-KI`.
3. Select it using `cd <index>`.
4. Run `train`.
5. Enter the requested number of generations and confirm the training run.

Example session:

```text
user/ > models
user/ > cd <Connect-Four-index>
user/Connect-Four-KI/ > train
Number of generations: 100
```

Use the index actually displayed by the application; it may change if the model configuration changes.

The model manager checks the selected model's `training_method` configuration. For models configured with `training_method: "evolutionary"`, `train` invokes the evolutionary training workflow rather than the standard gradient-based training options.

The Connect Four implementation evaluates candidate networks by playing simulated games against other networks. The current implementation uses a population-based selection process with parent selection and mutation to create subsequent generations.

Evolutionary training does not use the standard model's epoch, batch-size, learning-rate, validation, and checkpoint options. Its primary training parameter is the number of generations.

### Play manually against the saved network

To start a manual game, select the Connect Four model and run `use`:

```text
user/ > models
user/ > cd <Connect-Four-index>
user/Connect-Four-KI/ > use
```

The Connect Four prediction workflow launches the interactive console game. The game loads its saved network from `models/connect_four_model.json`. The human enters a column index from `0` to `6`; the board has six rows and seven columns.

The game and evolutionary training use the same Connect Four implementation, but simulated training games do not print every board. The manual game continues to display the board so the player can follow the moves.

**Important:** Playing against the saved network is a qualitative check, not a reliable benchmark of strength. Evaluate improvements across multiple games and compare results under consistent conditions before drawing conclusions about training progress.

## Dataset generation

Dataset-generation scripts are located in `scripts/`. Run them as modules from the repository root.

| Script | Purpose |
|---|---|
| `scripts.circle_dataset_generator` | Generate a three-class radial point-classification dataset |
| `scripts.prime_dataset_generator` | Generate balanced prime/non-prime classification examples |
| `scripts.odd_even_binary_dataset_generator` | Generate odd/even examples encoded as binary input bits |
| `scripts.odd_or_even_dataset_generator` | Generate odd/even examples using numeric inputs |
| `scripts.tic_tac_toe_training` | Generate Tic-Tac-Toe training data |

Examples:

```bash
python -m scripts.circle_dataset_generator
python -m scripts.prime_dataset_generator
python -m scripts.odd_even_binary_dataset_generator
python -m scripts.odd_or_even_dataset_generator
python -m scripts.tic_tac_toe_training
```

Some generators ask for parameters such as the number of examples, numeric limit, or output filename. Generated datasets are written to `data/`.

Connect Four is different: its evolutionary training simulates games directly, so its model configuration does not specify a conventional training dataset file.

## Training-data format

Standard model datasets are JSON arrays of examples. Each example contains an `inputs` list and a `targets` list, for example:

```json
[
  {
    "inputs": [0.25, -0.5, 0.3125],
    "targets": [1, 0, 0]
  },
  {
    "inputs": [0.7, 0.1, 0.5],
    "targets": [0, 1, 0]
  }
]
```

The input and target lengths must match the model's configured input and output dimensions. Training data is validated for finite numeric values, matching dimensions, non-negative target values, and target distributions that sum to 1.

The existing circle-classification model has a supported two-coordinate input form in addition to its configured three-feature representation.

## Project structure

```text
.
├── config/
│   ├── model_configs.json
│   ├── training_preferences.json
│   └── training_history.json       # Generated locally; ignored by Git
├── data/                           # Training datasets
├── models/                         # Saved model weights
├── scripts/                        # Dataset generators
├── tests/                          # unittest test suite
├── main.py                         # Application entry point
├── model_manager.py                # Public facade and entry point
├── model_manager_models.py         # Model workflows and prediction
├── model_manager_shell.py          # Interactive CLI and command handlers
├── model_manager_training.py       # Training configuration and helpers
├── neural_network.py               # Neural network implementation
├── tic_tac_toe.py                  # Tic-Tac-Toe game logic
├── connect_four.py                 # Connect Four game and evolution
├── terminal_colors.py              # Terminal formatting
├── requirements.txt                # Python dependencies
├── ruff.toml                       # Ruff lint configuration
└── .gitignore
```

The configuration file defines each model's dataset, saved weights, architecture, input handling, output labels, and training method where applicable. Training preferences and history are maintained separately from the model configurations.

## Tests and code quality

Run the automated tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The suite uses Python's built-in `unittest` framework. It covers training behavior and supporting project functionality.

Ruff is configured for linting, but it is not listed as a runtime dependency in `requirements.txt`. If Ruff is installed, run:

```bash
ruff check .
```

## Current scope and limitations

- This is an experimental learning project, not a production-ready machine-learning framework.
- The model manager supports the workflows implemented by each model configuration; not every model supports every training method.
- Connect Four uses evolutionary training, while standard classification models use the project's normal training pipeline.
- Game performance has to be measured empirically. A successful training run or a single win does not establish general playing strength.
- The project does not currently document a stable public package API or a release/versioning policy.

## License

No license file is currently included in the repository. Unless a license is added, users should not assume that the project is available for unrestricted reuse or redistribution.
