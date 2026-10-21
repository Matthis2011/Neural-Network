"""Generate balanced prime/non-prime classification examples."""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIRST_PRIME = 2
MINIMUM_PRIME_DATASET_LIMIT = 3


def is_prime(number):
    """Return whether a whole number is prime."""
    if number < FIRST_PRIME:
        return False

    return all(number % divisor != 0 for divisor in range(2, int(number**0.5) + 1))


def generate_dataset(how_many, limit):
    """Build a shuffled dataset with equal numbers of each class."""
    if how_many % 2 != 0:
        raise ValueError("The number of examples must be even.")
    if limit < MINIMUM_PRIME_DATASET_LIMIT:
        raise ValueError("The limit must be at least 3.")

    primes = []
    non_primes = []

    for number in range(FIRST_PRIME, limit + 1):
        if is_prime(number):
            primes.append(number)
        else:
            non_primes.append(number)

    dataset = []
    amount_per_class = how_many // 2

    for _ in range(amount_per_class):
        prime_number = random.choice(primes)
        non_prime_number = random.choice(non_primes)
        dataset.append({"inputs": [prime_number], "targets": [0, 1]})
        dataset.append({"inputs": [non_prime_number], "targets": [1, 0]})

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
