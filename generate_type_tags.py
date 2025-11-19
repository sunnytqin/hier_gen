"""
Helper functions for loading question formation data and converting tokens to types.
Extracted from generate_data_from_cfg.py for standalone use in hier_gen/.
"""
import pandas as pd
from collections import defaultdict


def load_question_formation_data(
    split, data_dir, include_identity=True, filename=None
):
    """
    Load question formation data from file.

    Args:
        split: 'train', 'val', or 'test'
        data_dir: Directory containing the question formation data
        include_identity: Whether to include declarative (identity) sentences
        filename: Optional custom filename (default: 'question')

    Returns:
        pandas DataFrame with 'input' and 'output' columns
    """
    if filename is None:
        filename = f"{data_dir}/question.{split}"
    else:
        filename = f"{data_dir}/{filename}.{split}"

    with open(filename) as f:
        lines = f.read().split("\n")
        if lines[-1] == "":
            lines = lines[:-1]

    in_sents, out_sents = [], []
    counts = {
        'quest': 0,
        'decl': 0
    }
    for line in lines:
        in_sent, out_sent = line.split("\t")

        if 'quest' in line:
            counts['quest'] += 1
        else:
            counts['decl'] += 1

        if not include_identity and "quest" not in in_sent:
            continue

        # Remove the task token (quest/decl) from end of input
        in_sent = " ".join(in_sent.split()[:-1])

        in_sents.append(in_sent)
        out_sents.append(out_sent)

    print(f"{split}: {counts}")

    return pd.DataFrame({"input": in_sents, "output": out_sents})


def token_seq_to_type_seq(token_seq, token2tag):
    """
    Convert a sequence of tokens to a sequence of types using the token-to-tag mapping.

    Args:
        token_seq: Space-separated string of tokens
        token2tag: Dictionary mapping tokens to their types

    Returns:
        Space-separated string of types
    """
    return " ".join(
        [token2tag.get(token, token) for token in token_seq.split()]
    ).strip()


def load_token_to_tag_mapping(cfg_dir):
    """
    Load the mapping from tokens to their grammatical type tags.

    Args:
        cfg_dir: Directory containing tag_token_map.txt

    Returns:
        tuple: (type_to_token_map dict, token2tag dict)
    """
    type_to_token_map = defaultdict(list)
    token2tag = {}

    with open(f"{cfg_dir}/tag_token_map.txt") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            type_, token = line.split("\t")
            type_to_token_map[type_].append(token)
            # Special corner case: "read" belongs to v_p_intrans for question_formation
            if token not in token2tag:
                token2tag[token] = type_

    # Punctuation doesn't have a type
    token2tag["."] = ""
    token2tag["?"] = ""

    return type_to_token_map, token2tag


def determine_qtype(input_tokens, output_tokens):
    """
    Determine the type of question transformation.

    Args:
        input_tokens: List of input tokens
        output_tokens: List of output tokens

    Returns:
        str: 'decl' (declarative), 'linear', or 'hier' (hierarchical)
    """
    auxs = ["doesn't", "does", "do", "don't"]

    # If input equals output, it's declarative (identity)
    if input_tokens == output_tokens:
        return "decl"

    # Find the first auxiliary in input (declarative)
    aux_decl = None
    for w in input_tokens:
        if w in auxs:
            aux_decl = w
            break

    # Find the first auxiliary in output (question)
    aux_quest = None
    for w in output_tokens:
        if w in auxs:
            aux_quest = w
            break

    # If auxiliaries are the same, it's linear; otherwise hierarchical
    if aux_decl == aux_quest:
        return "linear"
    else:
        return "hier"
