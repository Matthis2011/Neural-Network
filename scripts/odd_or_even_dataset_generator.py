"""Generate numeric odd/even classification examples."""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def generate_point(limit):
    """Generate one random integer and its odd/even target."""
    number = random.randint(1, limit)
    target = [0, 1] if number % 2 == 0 else [1, 0]
    return {"inputs": [number], "targets": target}


def generate_dataset(how_many, limit):
    """Build and shuffle a dataset with the requested number of examples."""
    dataset = []
    for _ in range(how_many):
        dataset.append(generate_point(limit))

    random.shuffle(dataset)

    return dataset


def save_dataset(data, filename):
    """Write generated examples to the project's data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def main():
    """Prompt for dataset settings and write the generated examples."""
    how_many = int(input("Number of examples to generate: "))
    limit = int(input("Maximum number value: "))
    name = input("Output filename: ").strip()

    data = generate_dataset(how_many, limit)
    save_dataset(data, name)


if __name__ == "__main__":
    main()
