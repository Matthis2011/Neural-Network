"""Generate binary-encoded odd/even classification examples."""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def number_to_bits(number, bit_count):
    """Represent a non-negative integer as a fixed-width bit list."""
    return [int(bit) for bit in f"{number:0{bit_count}b}"]


def generate_dataset(limit):
    """Build shuffled odd/even examples from one through the given limit."""
    if limit < 1:
        raise ValueError("The limit must be at least 1.")

    bit_count = limit.bit_length()
    dataset = []
    for number in range(1, limit + 1):
        dataset.append(
            {
                "inputs": number_to_bits(number, bit_count),
                "targets": [1, 0] if number % 2 else [0, 1],
            }
        )

    random.shuffle(dataset)
    return dataset, bit_count


def save_dataset(data, filename):
    """Write generated examples to the project's data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def main():
    """Prompt for a limit and save the generated dataset."""
    limit = int(input("Generate examples up to which number? "))
    name = input("Output filename: ").strip()
    data, bit_count = generate_dataset(limit)
    save_dataset(data, name)
    print(f"{len(data)} Datenpunkte mit {bit_count} Bits gespeichert.")


if __name__ == "__main__":
    main()
