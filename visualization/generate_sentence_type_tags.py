"""
Generate sentence type tag files for question formation data.

This script reads question formation data files and generates corresponding .type files
that tag each sentence with its grammatical type structure.

Usage:
    python generate_sentence_type_tags.py --split test
    python generate_sentence_type_tags.py --split train --filename question_D20

The script will create files like:
    - question.test.type
    - question_D20.train.type

Author: Adapted from visualization/analyze_data_from_cfg.py
"""
import sys
import os
from collections import defaultdict

# Add parent directory to path to import helper functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_type_tags import (
    load_question_formation_data,
    token_seq_to_type_seq,
    load_token_to_tag_mapping
)

# Configuration
CFG_DIR = "../cfgs"  # Relative to hier_gen/visualization/
DATA_DIR = "../data_utils/question_formation_data"  # Relative to hier_gen/visualization/


def generate_type_file(split="test", filename=None):
    """
    Generate a .type file for the given split and filename.

    Args:
        split: Data split to process ('train', 'val', or 'test')
        filename: Base filename (default: 'question')
    """
    if filename is None:
        filename = 'question'

    print(f"Generating type tags for {filename}.{split}")

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load token-to-tag mapping
    cfg_dir = os.path.join(script_dir, CFG_DIR)
    type_to_token_map, token2tag = load_token_to_tag_mapping(cfg_dir)

    # Load question formation data
    data_dir = os.path.join(script_dir, DATA_DIR)
    quest_form_data = load_question_formation_data(
        split=split,
        data_dir=data_dir,
        include_identity=True,
        filename=filename if filename != 'question' else None
    )

    # Convert token sequences to type sequences
    token_seq_to_type_seq_fn = lambda token_seq: token_seq_to_type_seq(
        token_seq, token2tag
    )

    input_types = (
        quest_form_data["input"]
        .apply(token_seq_to_type_seq_fn)
        .values.tolist()
    )

    output_types = (
        quest_form_data["output"]
        .apply(token_seq_to_type_seq_fn)
        .values.tolist()
    )

    print(f"Converted {len(input_types)} sentences to type sequences")

    # Save type file
    output_filename = f"{filename}.{split}.type"
    output_path = os.path.join(data_dir, output_filename)

    with open(output_path, "w") as f:
        for inp, out in zip(input_types, output_types):
            # Determine if this is a question or declarative
            task_type = "quest" if out.startswith("aux") else "decl"
            f.write(f"{inp} {task_type}\t{out}\n")

    print(f"Type file saved to: {output_path}")
    print(f"Total sentences tagged: {len(input_types)}")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate sentence type tag files for question formation data"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to process (default: test)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Base filename (default: 'question'). E.g., 'question_D20' for question_D20.train"
    )

    args = parser.parse_args()

    generate_type_file(split=args.split, filename=args.filename)


if __name__ == "__main__":
    main()
