#!/usr/bin/env python3
"""
Convert BenchForm BBH data to JSONL format.

This script converts the BBH validation data from BenchForm to JSONL format,
which is a common format for ML evaluation frameworks.

Usage:
    python convert_to_jsonl.py --output_dir ./jsonl_data

Output format (one JSON per line):
{
    "task": "causal_judgment",
    "idx": 0,
    "question": "Question text with answer choices...",
    "choices": ["No", "Yes"],
    "answer": "A",
    "answer_text": "No"
}
"""

import argparse
import json
import os
from pathlib import Path
from string import ascii_uppercase


# BBH tasks used in BenchForm
BBH_TASKS = [
    "sports_understanding",
    "snarks",
    "disambiguation_qa",
    "movie_recommendation",
    "causal_judgment",
    "date_understanding",
    "tracking_shuffled_objects_three_objects",
    "temporal_sequences",
    "ruin_names",
    "web_of_lies",
    "navigate",
    "logical_deduction_five_objects",
    "hyperbaton",
]


def convert_task_data(
    data_dir: Path,
    output_dir: Path,
    task: str,
) -> int:
    """
    Convert a single task's data to JSONL format.

    Returns the number of examples converted.
    """
    input_file = data_dir / task / "val_data.json"
    output_file = output_dir / f"{task}.jsonl"

    if not input_file.exists():
        print(f"Warning: {input_file} not found, skipping")
        return 0

    with open(input_file, "r") as f:
        data = json.load(f)

    examples = data.get("data", [])
    count = 0

    with open(output_file, "w") as f:
        for i, example in enumerate(examples):
            # Find the correct answer index
            correct_idx = example["multiple_choice_scores"].index(1)
            correct_letter = ascii_uppercase[correct_idx]

            # Create simplified record
            record = {
                "task": task,
                "idx": example.get("idx", i),
                "question": example["parsed_inputs"],
                "choices": example["multiple_choice_targets"],
                "answer": correct_letter,
                "answer_text": example["multiple_choice_targets"][correct_idx],
                # Keep original fields for compatibility
                "inputs": example.get("inputs", ""),
                "targets": example.get("targets", []),
                "multiple_choice_targets": example["multiple_choice_targets"],
                "multiple_choice_scores": example["multiple_choice_scores"],
            }

            f.write(json.dumps(record) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BenchForm BBH data to JSONL format"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/bbh",
        help="Path to BBH data directory (default: ./data/bbh)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./jsonl_data",
        help="Output directory for JSONL files (default: ./jsonl_data)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=BBH_TASKS,
        help="Tasks to convert (default: all tasks)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_count = 0
    for task in args.tasks:
        count = convert_task_data(data_dir, output_dir, task)
        print(f"Converted {task}: {count} examples")
        total_count += count

    print(f"\nTotal examples converted: {total_count}")
    print(f"Output directory: {output_dir}")

    # Create combined file
    combined_file = output_dir / "all_tasks.jsonl"
    with open(combined_file, "w") as outf:
        for task in args.tasks:
            task_file = output_dir / f"{task}.jsonl"
            if task_file.exists():
                with open(task_file, "r") as inf:
                    outf.write(inf.read())

    print(f"Combined file: {combined_file}")


if __name__ == "__main__":
    main()
