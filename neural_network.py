"""Implement a fully connected neural network and its training routine."""

import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass

from terminal_colors import colorize

PERCENTAGE_SCALE = 100
CIRCLE_COORDINATE_COUNT = 2
CIRCLE_FEATURE_INPUT_SIZE = 3
TRAINING_PROGRESS_INTERVAL_SECONDS = 5.0
MIN_VALIDATION_EXAMPLES = 2


@dataclass(frozen=True)
class LossReductionTarget:
    """Describe an absolute or percentage loss-reduction goal."""

    absolute: float | None = None
    percentage: float | None = None


@dataclass(frozen=True)
class TrainingSettings:
    """Collect optimization, validation, reproducibility, and checkpoint settings."""

    epochs: int
    batch_size: int
    learning_rate: float
    max_duration: float | None = None
    loss_reduction_target: LossReductionTarget | None = None
    verbose: bool = False
    validation_split: float = 0.0
    patience: int = 10
    min_delta: float = 0.0
    seed: int | None = None
    checkpoint_path: str | None = None
    resume_from: str | None = None
    dataset_hash: str | None = None


def sigmoid(x):
    """Calculate the logistic activation for a scalar input."""
    y = 1 / (1 + math.exp(-x))
    return y


def loss(predictions, targets):
    """Calculate categorical cross-entropy for class probabilities."""
    total = 0

    for i in range(len(predictions)):
        prediction = max(predictions[i], 1e-15)
        total += targets[i] * math.log(prediction)
    return -total


def softmax(inputs):
    """Convert logits into a numerically stable probability distribution."""
    max_value = inputs[0]
    after_softmax = []
    divisor = 0
    for value in inputs:
        max_value = max(max_value, value)
    for value in inputs:
        divisor += math.exp(value - max_value)
    for value in inputs:
        after_softmax.append(math.exp(value - max_value) / divisor)
    return after_softmax


def argmax(inputs):
    """Return the index of the largest value."""
    max_value = inputs[0]
    max_value_index = 0
    for i in range(len(inputs)):
        if inputs[i] > max_value:
            max_value = inputs[i]
            max_value_index = i
    return max_value_index


def get_batches(training_data, batch_size):
    """Yield consecutive batches from the training examples."""
    for i in range(0, len(training_data), batch_size):
        batch = training_data[i : i + batch_size]
        yield batch


def _nested_lists_to_tuples(value):
    """Restore tuple nesting required by Python's random-state API."""
    if isinstance(value, list):
        return tuple(_nested_lists_to_tuples(item) for item in value)
    return value


def _json_dataset_hash(training_data):
    """Return a stable identity for a dataset passed directly to the network."""
    encoded_data = json.dumps(
        training_data, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded_data).hexdigest()


class Neuron:
    """Represent a single weighted unit in a neural network layer."""

    def __init__(self, input_neurons, bias, is_output_layer, rng=None):
        """Initialize weights and activation state for a neuron."""
        limit = math.sqrt(6 / (input_neurons + 1))
        random_source = random if rng is None else rng
        self.weights = [
            random_source.uniform(-limit, limit) for _ in range(input_neurons)
        ]
        self.bias = bias
        self.is_output_layer = is_output_layer
        self.z = 0
        self.activation = 0

    def forward(self, inputs):
        """Compute and store this neuron's activation for the inputs."""
        total = 0

        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i]

        self.z = total + self.bias
        output = sigmoid(self.z) if not self.is_output_layer else self.z
        self.activation = output
        return output


class Layer:
    """Group neurons and compute forward and backward layer operations."""

    def __init__(self, input_size, neuron_count, is_output_layer, rng=None):
        """Create a layer with the requested number of neurons."""
        neurons = []
        for _ in range(neuron_count):
            neurons.append(Neuron(input_size, 0, is_output_layer, rng))
        self.neurons = neurons
        self.activations = []
        self.is_output_layer = is_output_layer

    def forward(self, inputs):
        """Return the activations produced by all neurons in the layer."""
        outputs = []

        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))

        self.activations = outputs

        return outputs

    def backward(self, targets, next_layer, next_gradients, previous_activations):
        """Calculate bias and weight gradients for the layer."""
        gradients = []
        weight_gradients = []
        probabilities = []
        if self.is_output_layer:
            probabilities = softmax(self.activations)

        for i in range(len(self.neurons)):
            gradient = 0

            if not self.is_output_layer:
                for j in range(len(next_layer.neurons)):
                    gradient += next_layer.neurons[j].weights[i] * next_gradients[j]
                gradient *= self.neurons[i].activation * (
                    1 - self.neurons[i].activation
                )
            else:
                gradient = probabilities[i] - targets[i]

            gradients.append(gradient)
        for i in range(len(self.neurons)):
            neuron_weight_gradients = []

            for j in range(len(self.neurons[i].weights)):
                gradient = gradients[i] * previous_activations[j]
                neuron_weight_gradients.append(gradient)

            weight_gradients.append(neuron_weight_gradients)
        return gradients, weight_gradients


class NeuralNetwork:
    """A fully connected feed-forward classifier with gradient training."""

    def __init__(self, input_size, hidden_sizes, output_size, seed=None):
        """Build a network from input, hidden, and output layer sizes."""
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.hidden_layers = []
        rng = random if seed is None else random.Random(seed)
        previous_size = self.input_size
        for size in hidden_sizes:
            self.hidden_layers.append(Layer(previous_size, size, False, rng))
            previous_size = size
        self.output_layer = Layer(previous_size, output_size, True, rng)

    def forward(self, inputs):
        """Return the output logits for one input sample."""
        if (
            len(inputs) == CIRCLE_COORDINATE_COUNT
            and self.input_size == CIRCLE_FEATURE_INPUT_SIZE
        ):
            x, y = inputs
            inputs = [x, y, x * x + y * y]

        current = inputs
        for layer in self.hidden_layers:
            current = layer.forward(current)
        current = self.output_layer.forward(current)
        return current

    def training_step(self, inputs, targets):
        """Calculate gradients for one labeled input sample."""
        all_gradients: list[list[float]] = [
            [] for _ in range(len(self.hidden_layers) + 1)
        ]
        all_weight_gradients: list[list[list[float]]] = [
            [] for _ in range(len(self.hidden_layers) + 1)
        ]

        if (
            len(inputs) == CIRCLE_COORDINATE_COUNT
            and self.input_size == CIRCLE_FEATURE_INPUT_SIZE
        ):
            x, y = inputs
            inputs = [x, y, x * x + y * y]

        self.forward(inputs)

        gradients, weight_gradients = self.output_layer.backward(
            targets,
            None,
            None,
            (self.hidden_layers[-1].activations if self.hidden_layers else inputs),
        )

        all_gradients[-1] = gradients
        all_weight_gradients[-1] = weight_gradients

        next_gradients = gradients

        for i in range(len(self.hidden_layers) - 1, -1, -1):
            current_layer = self.hidden_layers[i]

            if i == len(self.hidden_layers) - 1:
                next_layer = self.output_layer
            else:
                next_layer = self.hidden_layers[i + 1]

            if i == 0:
                previous_activations = inputs
            else:
                previous_activations = self.hidden_layers[i - 1].activations

            gradients, weight_gradients = current_layer.backward(
                targets, next_layer, next_gradients, previous_activations
            )

            all_gradients[i] = gradients
            all_weight_gradients[i] = weight_gradients

            next_gradients = gradients

        return all_gradients, all_weight_gradients

    def save(self, filename):
        """Serialize this network's architecture and learned parameters."""
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4)

    def to_dict(self):
        """Return the architecture and learned parameters as JSON-safe data."""
        data = {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
            "hidden_layers": [],
            "output_layer": [],
        }

        for layer in self.hidden_layers:
            layer_data = []

            for neuron in layer.neurons:
                layer_data.append(
                    {"weights": list(neuron.weights), "bias": neuron.bias}
                )

            data["hidden_layers"].append(layer_data)

        for neuron in self.output_layer.neurons:
            data["output_layer"].append(
                {"weights": list(neuron.weights), "bias": neuron.bias}
            )

        return data

    def load_dict(self, data, strict_dimensions=True):
        """Restore parameters from a compatible serialized network."""
        architecture = (
            data.get("input_size"),
            data.get("hidden_sizes"),
            data.get("output_size"),
        )
        if architecture != (self.input_size, self.hidden_sizes, self.output_size):
            raise ValueError("Checkpoint model architecture does not match.")
        for layer, layer_data in zip(
            self.hidden_layers, data["hidden_layers"], strict=True
        ):
            if len(layer.neurons) != len(layer_data):
                raise ValueError("Checkpoint hidden-layer dimensions do not match.")
            for neuron, neuron_data in zip(layer.neurons, layer_data, strict=True):
                expected_width = len(neuron.weights)
                saved_weights = neuron_data["weights"]
                if len(saved_weights) != expected_width and (
                    strict_dimensions or len(saved_weights) < expected_width
                ):
                    raise ValueError("Saved neuron dimensions do not match.")
                if len(saved_weights) > expected_width:
                    saved_weights = saved_weights[:expected_width]
                neuron.weights = saved_weights
                neuron.bias = neuron_data["bias"]
        if len(self.output_layer.neurons) != len(data["output_layer"]):
            raise ValueError("Checkpoint output-layer dimensions do not match.")
        for neuron, neuron_data in zip(
            self.output_layer.neurons, data["output_layer"], strict=True
        ):
            expected_width = len(neuron.weights)
            saved_weights = neuron_data["weights"]
            if len(saved_weights) != expected_width and (
                strict_dimensions or len(saved_weights) < expected_width
            ):
                raise ValueError("Saved neuron dimensions do not match.")
            if len(saved_weights) > expected_width:
                saved_weights = saved_weights[:expected_width]
            neuron.weights = saved_weights
            neuron.bias = neuron_data["bias"]

    @classmethod
    def load(cls, filename):
        """Load a network and its learned parameters from a JSON file."""
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            input_size = data["input_size"]
            hidden_sizes = data["hidden_sizes"]
            output_size = data["output_size"]
            network = cls(input_size, hidden_sizes, output_size)
            network.load_dict(data, strict_dimensions=False)
        return network

    def _evaluate_loss(self, training_data):
        """Return mean loss over the complete training dataset."""
        total_loss = sum(
            loss(
                softmax(self.forward(data_point["inputs"])),
                data_point["targets"],
            )
            for data_point in training_data
        )
        return total_loss / len(training_data)

    def _train_epoch(self, training_data, batch_size, learning_rate):
        """Run one gradient epoch and return its pre-update average loss."""
        average_loss = self._evaluate_loss(training_data)
        layers = [*self.hidden_layers, self.output_layer]

        for batch in get_batches(training_data, batch_size):
            gradient_sums = [[0.0] * len(layer.neurons) for layer in layers]
            weight_gradient_sums = [
                [[0.0] * len(neuron.weights) for neuron in layer.neurons]
                for layer in layers
            ]
            for data_point in batch:
                gradients, weight_gradients = self.training_step(
                    data_point["inputs"], data_point["targets"]
                )
                for layer_index, layer_gradients in enumerate(gradients):
                    for neuron_index, gradient in enumerate(layer_gradients):
                        gradient_sums[layer_index][neuron_index] += gradient
                    for neuron_index, neuron_weights in enumerate(
                        weight_gradients[layer_index]
                    ):
                        for weight_index, gradient in enumerate(neuron_weights):
                            weight_gradient_sums[layer_index][neuron_index][
                                weight_index
                            ] += gradient

            for layer_index, layer in enumerate(layers):
                for neuron_index, neuron in enumerate(layer.neurons):
                    neuron.bias -= (
                        learning_rate
                        * gradient_sums[layer_index][neuron_index]
                        / len(batch)
                    )
                    for weight_index in range(len(neuron.weights)):
                        neuron.weights[weight_index] -= (
                            learning_rate
                            * weight_gradient_sums[layer_index][neuron_index][
                                weight_index
                            ]
                            / len(batch)
                        )
        return average_loss

    @staticmethod
    def _validate_training_settings(settings):
        """Reject invalid epoch, batch, time, or loss-target values."""
        if settings.epochs < 1:
            raise ValueError("Epoch count must be at least 1.")
        if settings.batch_size < 1:
            raise ValueError("Batch size must be at least 1.")
        if not math.isfinite(settings.learning_rate) or settings.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and greater than zero.")
        if settings.seed is not None and (
            not isinstance(settings.seed, int)
            or isinstance(settings.seed, bool)
            or settings.seed < 0
        ):
            raise ValueError("Random seed must be a non-negative integer.")
        if (
            not math.isfinite(settings.validation_split)
            or not 0 <= settings.validation_split < 1
        ):
            raise ValueError("Validation split must be in [0, 1).")
        if settings.validation_split and settings.patience < 1:
            raise ValueError("Early-stopping patience must be at least 1.")
        if not math.isfinite(settings.min_delta) or settings.min_delta < 0:
            raise ValueError(
                "Early-stopping min_delta must be finite and non-negative."
            )
        if settings.max_duration is not None and (
            not math.isfinite(settings.max_duration) or settings.max_duration <= 0
        ):
            raise ValueError("Time budget must be finite and greater than zero.")
        target = settings.loss_reduction_target
        if target is not None:
            if target.absolute is not None and (
                not math.isfinite(target.absolute) or target.absolute <= 0
            ):
                raise ValueError("Loss reduction must be a finite positive number.")
            if target.percentage is not None and (
                not math.isfinite(target.percentage)
                or not 0 < target.percentage <= PERCENTAGE_SCALE
            ):
                raise ValueError("Loss-reduction percentage must be in (0, 100].")

    @staticmethod
    def _write_checkpoint(filename, checkpoint):
        """Atomically write a versioned JSON training checkpoint."""
        path = os.fspath(filename)
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            dir=directory, prefix=".training-checkpoint-", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(checkpoint, file, allow_nan=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        except Exception:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise

    def _read_checkpoint(self, settings, data_hash, rng, data_size):
        """Load and validate saved epoch-boundary training state."""
        try:
            with open(settings.resume_from, "r", encoding="utf-8") as file:
                checkpoint = json.load(file)
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint root must be a JSON object.")
            if checkpoint.get("format_version") != 1:
                raise ValueError("Unsupported checkpoint format version.")
            if checkpoint.get("dataset_hash") != data_hash:
                raise ValueError("Checkpoint dataset does not match the current data.")
            if checkpoint.get("settings") != self._checkpoint_settings(settings):
                raise ValueError(
                    "Checkpoint training settings do not match. "
                    "Keep batch size, learning rate, validation options, seed, "
                    "and loss target unchanged when resuming."
                )
            training_indices = checkpoint["training_indices"]
            validation_indices = checkpoint["validation_indices"]
            all_indices = training_indices + validation_indices
            if (
                len(all_indices) != data_size
                or set(all_indices) != set(range(data_size))
                or len(set(all_indices)) != data_size
            ):
                raise ValueError("Checkpoint data split is invalid.")
            self.load_dict(checkpoint["network"])
            rng.setstate(_nested_lists_to_tuples(checkpoint["rng_state"]))
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
            OverflowError,
        ) as error:
            raise ValueError(f"Could not read training checkpoint: {error}") from error
        return checkpoint

    @staticmethod
    def _checkpoint_settings(settings):
        """Select settings that must remain fixed across a resumed run."""
        return {
            "batch_size": settings.batch_size,
            "learning_rate": settings.learning_rate,
            "validation_split": settings.validation_split,
            "patience": settings.patience,
            "min_delta": settings.min_delta,
            "seed": settings.seed,
            "loss_reduction_target": (
                {
                    "absolute": settings.loss_reduction_target.absolute,
                    "percentage": settings.loss_reduction_target.percentage,
                }
                if settings.loss_reduction_target is not None
                else None
            ),
        }

    @staticmethod
    def _accuracy(network, data):
        """Return classification accuracy on a non-empty dataset."""
        return sum(
            argmax(softmax(network.forward(point["inputs"])))
            == argmax(point["targets"])
            for point in data
        ) / len(data)

    def _initialize_training_state(self, data, settings, rng, data_hash):
        """Create a deterministic split or restore a compatible checkpoint."""
        if settings.resume_from is not None:
            checkpoint = self._read_checkpoint(settings, data_hash, rng, len(data))
            return {
                key: checkpoint[key]
                for key in (
                    "training_indices",
                    "validation_indices",
                    "epoch_losses",
                    "epoch_durations_seconds",
                    "validation_losses",
                    "best_validation_loss",
                    "best_validation_epoch",
                    "best_network",
                    "epochs_without_improvement",
                    "epochs_completed",
                    "initial_loss",
                )
            }

        indices = list(range(len(data)))
        validation_indices = []
        if settings.validation_split:
            rng.shuffle(indices)
            validation_count = max(1, round(len(indices) * settings.validation_split))
            validation_count = min(validation_count, len(indices) - 1)
            validation_indices = indices[:validation_count]
            indices = indices[validation_count:]
        return {
            "training_indices": indices,
            "validation_indices": validation_indices,
            "epoch_losses": [],
            "epoch_durations_seconds": [],
            "validation_losses": [],
            "best_validation_loss": None,
            "best_validation_epoch": None,
            "best_network": None,
            "epochs_without_improvement": 0,
            "epochs_completed": 0,
            "initial_loss": None,
        }

    def _update_validation_state(self, state, validation_examples, settings):
        """Track validation progress and retain the best parameter snapshot."""
        validation_loss = self._evaluate_loss(validation_examples)
        state["validation_losses"].append(validation_loss)
        best_loss = state["best_validation_loss"]
        if best_loss is None or validation_loss < best_loss - settings.min_delta:
            state["best_validation_loss"] = validation_loss
            state["best_validation_epoch"] = state["epochs_completed"]
            state["best_network"] = self.to_dict()
            state["epochs_without_improvement"] = 0
        else:
            state["epochs_without_improvement"] += 1

    def _save_training_checkpoint(self, settings, data_hash, rng, state):
        """Persist complete epoch-boundary state when checkpointing is enabled."""
        checkpoint_path = settings.checkpoint_path or settings.resume_from
        if checkpoint_path is None:
            return
        checkpoint = {
            "format_version": 1,
            "dataset_hash": data_hash,
            "settings": self._checkpoint_settings(settings),
            "network": self.to_dict(),
            "rng_state": rng.getstate(),
            **state,
        }
        self._write_checkpoint(checkpoint_path, checkpoint)

    @staticmethod
    def _print_training_progress(
        settings, average_loss, completed_epochs, total_epochs, progress_clock
    ):
        """Display epoch progress using TTY-aware updates."""
        started, last_print = progress_clock
        now = time.perf_counter()
        terminal_progress = sys.stdout.isatty() and not settings.verbose
        label = (
            f"Epoch {completed_epochs}"
            if settings.max_duration is not None
            else f"Epoch {completed_epochs}/{total_epochs}"
        )
        progress = f"{label} | Loss: {average_loss:.6f} | Elapsed: {now - started:.1f}s"
        if settings.verbose:
            print(colorize(progress, "accent"))
        elif terminal_progress:
            print(f"\r{colorize(progress, 'accent')}", end="", flush=True)
        elif now - last_print >= TRAINING_PROGRESS_INTERVAL_SECONDS:
            print(colorize(progress, "accent"))
            last_print = now
        progress_clock[1] = last_print

    def _run_training_epochs(self, context):
        """Run epochs, update validation state, and stop at requested limits."""
        settings = context["settings"]
        state = context["state"]
        training_data = context["training_data"]
        train_examples = context["train_examples"]
        validation_examples = context["validation_examples"]
        rng = context["rng"]
        training_started = context["training_started"]
        stopped_by_time = False
        target_reached = False
        stopped_by_early_stopping = False

        for epoch in range(settings.epochs):
            if (
                epoch > 0
                and settings.max_duration is not None
                and time.perf_counter() - training_started >= settings.max_duration
            ):
                stopped_by_time = True
                break
            epoch_started = time.perf_counter()
            rng.shuffle(state["training_indices"])
            train_examples[:] = [
                training_data[index] for index in state["training_indices"]
            ]
            average_loss = self._train_epoch(
                train_examples, settings.batch_size, settings.learning_rate
            )
            state["epoch_losses"].append(average_loss)
            state["epochs_completed"] += 1
            if validation_examples:
                self._update_validation_state(state, validation_examples, settings)
            self._print_training_progress(
                settings,
                average_loss,
                state["epochs_completed"],
                state["epochs_completed"] + settings.epochs - epoch - 1,
                context["progress_clock"],
            )
            state["epoch_durations_seconds"].append(time.perf_counter() - epoch_started)
            self._save_training_checkpoint(settings, context["data_hash"], rng, state)
            if settings.loss_reduction_target is not None:
                target_reached = self._target_is_reached(
                    state["initial_loss"],
                    self._evaluate_loss(training_data),
                    settings.loss_reduction_target,
                )
            if target_reached:
                break
            if (
                validation_examples
                and state["epochs_without_improvement"] >= settings.patience
            ):
                stopped_by_early_stopping = True
                break
            if (
                settings.max_duration is not None
                and time.perf_counter() - training_started >= settings.max_duration
            ):
                stopped_by_time = True
                break
        return stopped_by_time, target_reached, stopped_by_early_stopping

    @staticmethod
    def _target_is_reached(initial_loss, current_loss, target):
        """Check whether a measured loss satisfies the requested reduction."""
        if target is None:
            return False
        reduction = initial_loss - current_loss
        reduction_percent = (
            reduction / initial_loss * PERCENTAGE_SCALE if initial_loss else 0
        )
        return (target.absolute is not None and reduction >= target.absolute) or (
            target.percentage is not None and reduction_percent >= target.percentage
        )

    def train(self, training_data, settings: TrainingSettings):
        """Train with explicit epoch, optimizer, time, and loss-target settings."""
        self._validate_training_settings(settings)
        if not training_data:
            raise ValueError("Training data must contain at least one example.")
        if settings.validation_split and len(training_data) < MIN_VALIDATION_EXAMPLES:
            raise ValueError("Validation requires at least two examples.")
        training_started = time.perf_counter()
        rng = random if settings.seed is None else random.Random(settings.seed)
        data_hash = settings.dataset_hash or _json_dataset_hash(training_data)
        state = self._initialize_training_state(training_data, settings, rng, data_hash)
        train_examples = [training_data[index] for index in state["training_indices"]]
        validation_examples = [
            training_data[index] for index in state["validation_indices"]
        ]
        if not train_examples:
            raise ValueError("Training split must contain at least one example.")
        if state["initial_loss"] is None:
            state["initial_loss"] = self._evaluate_loss(training_data)
        stopped_by_time, target_reached, stopped_by_early_stopping = (
            self._run_training_epochs(
                {
                    "settings": settings,
                    "state": state,
                    "training_data": training_data,
                    "train_examples": train_examples,
                    "validation_examples": validation_examples,
                    "rng": rng,
                    "training_started": training_started,
                    "progress_clock": [training_started, training_started],
                    "data_hash": data_hash,
                }
            )
        )
        if sys.stdout.isatty() and not settings.verbose and state["epoch_losses"]:
            print()
        if validation_examples and state["best_network"] is not None:
            self.load_dict(state["best_network"])
        final_loss = self._evaluate_loss(training_data)
        final_validation_loss = (
            self._evaluate_loss(validation_examples) if validation_examples else None
        )
        validation_accuracy = (
            self._accuracy(self, validation_examples) if validation_examples else None
        )
        target_reached = (
            self._target_is_reached(
                state["initial_loss"], final_loss, settings.loss_reduction_target
            )
            if settings.loss_reduction_target is not None
            else False
        )
        correct_predictions = self._accuracy(self, train_examples)
        print(colorize(f"Final loss: {final_loss:.6f}", "success", bold=True))
        return {
            "epochs_completed": state["epochs_completed"],
            "epoch_losses": state["epoch_losses"],
            "epoch_durations_seconds": state["epoch_durations_seconds"],
            "validation_losses": state["validation_losses"],
            "validation_loss": final_validation_loss,
            "validation_accuracy": validation_accuracy,
            "best_validation_loss": state["best_validation_loss"],
            "best_validation_epoch": state["best_validation_epoch"],
            "initial_loss": state["initial_loss"],
            "final_loss": final_loss,
            "training_accuracy": correct_predictions,
            "dataset_size": len(training_data),
            "training_dataset_size": len(train_examples),
            "validation_dataset_size": len(validation_examples),
            "stopped_by_time": stopped_by_time,
            "stopped_by_early_stopping": stopped_by_early_stopping,
            "target_reached": target_reached,
            "duration_seconds": time.perf_counter() - training_started,
        }


def show_predictions(network, points, expected):
    """Print predictions and the accuracy for a small set of test points."""
    correct = 0

    for point, target in zip(points, expected, strict=False):
        raw_output = network.forward(point)
        probabilities = softmax(raw_output)
        prediction = argmax(probabilities)

        if prediction == target:
            correct += 1

        print(
            f"Punkt {point} | "
            f"Soll: Klasse {target} | "
            f"KI: Klasse {prediction} | "
            f"Wahrscheinlichkeiten: "
            f"[{probabilities[0]:.2f}, "
            f"{probabilities[1]:.2f}, "
            f"{probabilities[2]:.2f}]"
        )

    accuracy = correct / len(points) * 100

    print(f"\nTreffer: {correct}/{len(points)} ({accuracy:.1f} %)")


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        project_dir, "data", "circle_point_classification_training.json"
    )
    with open(path, "r", encoding="utf-8") as file:
        training_data = json.load(file)
    network = NeuralNetwork(input_size=3, hidden_sizes=[8, 8], output_size=3)
    test_points = [
        [0.0, 0.0],
        [0.2, 0.1],
        [0.5, 0.0],
        [0.6, 0.0],
        [0.8, 0.2],
        [-0.3, 0.3],
        [-0.6, 0.1],
        [0.0, -0.7],
        [0.4, -0.4],
        [-0.4, -0.5],
    ]
    expected = [0, 0, 1, 1, 2, 0, 1, 1, 1, 2]
    print("=== VOR DEM TRAINING ===")
    show_predictions(network, test_points, expected)

    print("\n=== TRAINING ===")

    network.train(
        training_data,
        TrainingSettings(epochs=1000, batch_size=32, learning_rate=0.1),
    )

    model_path = os.path.join(
        project_dir, "models", "circle_point_classification_model.json"
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    network.save(model_path)
    print(f"Modell gespeichert: {model_path}")

    print("\n=== NACH DEM TRAINING ===")
    show_predictions(network, test_points, expected)

    loaded_network = NeuralNetwork.load(model_path)
    print("\n=== AUS GESPEICHERTEM MODELL GELADEN ===")
    show_predictions(loaded_network, test_points, expected)
