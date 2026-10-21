"""Generate a balanced three-class radial point-classification dataset."""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INNER_CLASS_RADIUS_SQUARED = 0.2
MIDDLE_CLASS_RADIUS_SQUARED = 0.5


def generate_point():
    """Generate one point and assign its class from the squared radius."""
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    r_squared = x**2 + y**2
    if r_squared < INNER_CLASS_RADIUS_SQUARED:
        target = [1, 0, 0]
    elif r_squared < MIDDLE_CLASS_RADIUS_SQUARED:
        target = [0, 1, 0]
    else:
        target = [0, 0, 1]
    return {"inputs": [x, y, r_squared], "targets": target}


def generate_dataset(how_many):
    """Generate examples with approximately equal counts in each class."""
    dataset = []

    class_ranges = [
        (0.0, INNER_CLASS_RADIUS_SQUARED, 0),
        (INNER_CLASS_RADIUS_SQUARED, MIDDLE_CLASS_RADIUS_SQUARED, 1),
        (MIDDLE_CLASS_RADIUS_SQUARED, 2.0, 2),
    ]
    points_per_class = how_many // len(class_ranges)
    remainder = how_many % len(class_ranges)

    for class_index, (minimum, maximum, target_class) in enumerate(class_ranges):
        target_count = points_per_class + (class_index < remainder)
        class_count = 0

        while class_count < target_count:
            point = generate_point()
            radius = point["inputs"][2]
            if minimum <= radius < maximum:
                point["targets"] = [0, 0, 0]
                point["targets"][target_class] = 1
                dataset.append(point)
                class_count += 1

    random.shuffle(dataset)

    return dataset


def save_dataset(data, filename):
    """Write generated examples to the project's data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def main():
    """Prompt for dataset size and write the generated examples."""
    how_many = int(input("Number of examples to generate: "))
    name = input("Output filename: ").strip()

    data = generate_dataset(how_many)
    save_dataset(data, name)


if __name__ == "__main__":
    main()
