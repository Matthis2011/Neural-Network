# Neural Network Model Manager

A small NumPy-based neural-network project with an interactive console manager,
several training datasets, saved model weights, and dataset-generation utilities.

## Requirements

- Python 3.10 or newer
- NumPy (install with `python -m pip install -r requirements.txt`)

## Start the application

From this directory, run:

```bash
python main.py
```

The manager lists configured models and opens a shell-like prompt. Use `help`
for the concise command summary, `cd <index>` to enter a model, and
`help --train` for detailed command usage. For example:

```text
user/ > models
user/ > cd 1
user/Kreis-Klassifikation/ > stats
user/Kreis-Klassifikation/ > stats --detailed
user/Kreis-Klassifikation/ > train
user/Kreis-Klassifikation/ > train --t=10m
user/Kreis-Klassifikation/ > train --reduce-percent=20
user/Kreis-Klassifikation/ > train --e=500 --validation-split=0.2 --patience=12 --seed=42
user/Kreis-Klassifikation/ > train --e=100 --checkpoint=checkpoints/circle.json
user/Kreis-Klassifikation/ > train --e=100 --resume=checkpoints/circle.json
user/Kreis-Klassifikation/ > help --history
```

Training prompts for confirmation unless `-y` / `--yes` is supplied. Preferences
and training history are kept per model in `config/`. Running `train` without
arguments opens an interactive setup with that model's saved epoch, batch-size,
and learning-rate settings; enter replacement options or press Enter to accept
the suggested settings. If no epoch limit has been saved yet, enter an objective
such as `--e=500` or `--t=10m`. Explicit training arguments continue to run
directly with the normal preview and confirmation. Loss targets can be
specified as an absolute reduction (`train --r=0.15`) or a percentage
(`train --reduce-percent=20`). Options support short forms such as `-e 100`
and double-dash assignment forms such as `--e=100`; descriptive long names
such as `--epochs=100` are also accepted. Boolean options use `-v` or
`--verbose`. Target runs check measured full-dataset loss at
each epoch boundary and stop when the target is reached. You can combine a
target with `-e` and/or `-t` to set hard limits; target-only runs use previous
comparable training curves to estimate a safe epoch cap, or 1,000 epochs when
there is no matching history. Time budgets are checked at epoch boundaries,
and final loss/accuracy evaluation can add a little time afterward.
Training progress refreshes on one line in an interactive terminal and is
reported periodically when output is redirected. Add `-v` / `--verbose` to
print the loss for every epoch. Use `help --<command>` for detailed
syntax and examples; `help` alone stays concise.

Before training, the manager validates every example for the configured input
and output dimensions, finite numeric values, and a non-negative target
distribution that sums to 1. The existing three-feature circle network also
accepts its supported two-coordinate input form. To enable validation metrics
and early stopping, pass `--validation-split=0.2` (20% holdout);
`--patience=10` controls how many
epochs without validation-loss improvement are allowed. The split is
reproducible with `--seed=42`; for newly created models the same seed also
controls initial weights. The best validation weights are restored at the end,
and validation loss/accuracy and the best epoch appear in `history` and
`stats --detailed`. Validation is disabled by default to preserve existing
training behavior.

Use `--checkpoint=<file>` to atomically save a versioned JSON checkpoint at
every epoch boundary. The `create` workflow offers the same feature through
interactive prompts and asks before resuming an existing checkpoint. Resume
with `--resume=<file>`; the requested `-e` count is additional epochs.
Checkpoints preserve model weights, data split, random
state, validation progress, and epoch metrics. Resume requires the same
dataset, batch size, learning rate, validation split, patience, seed, and
loss-reduction target; time or epoch budgets may be changed. A resumed run
updates its checkpoint in place unless another `--checkpoint` path is supplied.
Run history records the seed and source dataset's SHA-256 digest for traceability.

## Project layout

```text
.
├── config/       Model configurations and saved training preferences/history
├── data/         Training datasets
├── models/       Saved model weights
├── scripts/      Dataset generators and Tic-Tac-Toe training-data generator
├── tests/        Automated tests
├── main.py       Application entry point
├── model_manager.py
├── neural_network.py
├── tic_tac_toe.py
└── connect_four.py
```

Generate a dataset with one of the scripts in `scripts/`, for example:

```bash
python -m scripts.tic_tac_toe_training
python -m scripts.prime_dataset_generator
python -m scripts.odd_even_binary_dataset_generator
python -m scripts.odd_or_even_dataset_generator
python -m scripts.circle_dataset_generator
```

The generators write output into `data/`.

## Tests

Run the standard-library test suite from this directory:

```bash
python -m unittest discover -s tests -v
```
