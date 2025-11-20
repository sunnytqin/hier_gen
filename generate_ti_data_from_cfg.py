import os
import copy
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm
from vocabulary import WordVocabulary

"""
This script generates synthetic tense inflection training data according to the original
dataset distribution by replicating sentence types and randomly sampling tokens.

The script supports:
- Generating PAST tense data (identity mapping: past -> past)
- Generating PRESENT tense data (inflection: past -> present with subject-verb agreement)
- Controlling the ratio of PAST/PRESENT samples
- Controlling the ratio of different sentence types (V, PREP, RC)
- Controlling the ratio of different RC subtypes (subject vs object relative clauses)
"""

# Directory paths (relative to hier_gen/)
CFG_DIR = "cfgs/"  # Contains tag_token_map.txt for type-to-token mappings
QUES_FORM_DATA_DIR = "data_utils/tense_inflection_data"  # Source data directory
DATA_DIR = "data_utils/"  # Output directory for generated data


def load_tense_inflection_data(
    split, include_identity=True, include_present=True, filename=None
):
    """
    Load tense inflection data from file and filter based on parameters.

    Args:
        split: Data split ('train', 'val', 'test')
        include_identity: Whether to include PAST->PAST identity mappings
        include_present: Whether to include PAST->PRESENT inflections
        filename: Optional custom filename (defaults to tense.{split})

    Returns:
        DataFrame with 'input' and 'output' columns
    """
    if filename is None:
        filename = f"{QUES_FORM_DATA_DIR}/tense.{split}"
    else:
        filename = f"{QUES_FORM_DATA_DIR}/{filename}.{split}"

    with open(filename) as f:
        lines = f.read().split("\n")
        if lines[-1] == "":
            lines = lines[:-1]

    in_sents, out_sents = [], []
    counts = {
        'PAST': 0,
        'PRESENT': 0
    }

    # Parse each line and filter based on parameters
    for line in lines:
        in_sent, out_sent = line.split("\t")

        if 'PRESENT' in line: counts['PRESENT'] += 1
        else: counts['PAST'] += 1

        # Filter based on include parameters
        if not include_identity and "PAST" in in_sent:
            continue
        if not include_present and "PRESENT" in in_sent:
            continue

        # Remove the tense marker (last token) from input
        in_sent = " ".join(in_sent.split()[:-1])

        in_sents.append(in_sent)
        out_sents.append(out_sent)
    print(f"{split}: {counts}")

    return pd.DataFrame({"input": in_sents, "output": out_sents})


def token_seq_to_type_seq(token_seq, token2tag):
    """
    Convert a sequence of tokens to their corresponding type tags.

    Example: "the dog giggled ." -> "det n_s v_past_intrans"
    """
    return " ".join(
        [token2tag.get(token, token) for token in token_seq.split()]
    ).strip()


# Mapping from past tense verbs to [plural present, singular present] forms
# Used to generate present tense output from past tense input
tense_map = {
    "giggled" : ["giggle", "giggles"],
    "smiled" : ["smile", "smiles"],
    "slept": ["sleep", "sleeps"],
    "swam": ["swim", "swims"],
    "waited": ["wait", "waits"],
    "moved":  ["move", "moves"],
    "changed": ["change", "changes"],
    "read": ["read", "reads"],
    "ate": ["eat", "eats"],
    "entertained": ["entertain", "entertains"],
    "amused": ["amuse", "amuses"],
    "high_fived": ["high_five", "high_fives"],
    "applauded": ["applaud", "applauds"],
    "confused": ["confuse", "confuses"],
    "admired": ["admire", "admires"],
    "accepted": ["accept", "accepts"],
    "remembered": ["remember", "remembers"],
    "comforted": ["comfort", "comforts"],
}


def main(args):
    np.random.seed(args.seed)

    # ========== STEP 1: Load Type-to-Token Mappings ==========
    # These mappings allow us to:
    # 1. Map types (e.g., "det", "n_s", "v_past_intrans") to tokens (e.g., "the", "dog", "giggled")
    # 2. Map tokens back to their types for identifying sentence structure

    type_to_token_map_present = defaultdict(list)  # type -> [tokens] for present tense
    type_to_token_map_past = defaultdict(list)     # type -> [tokens] for past tense
    token2tag_present = {}  # token -> type for present tense
    token2tag_past = {}     # token -> type for past tense

    with open(f"{CFG_DIR}/tag_token_map.txt") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            type_, token = line.split("\t")

            # Verbs are handled differently for past vs present
            if type_[0] == "v":
                if "past" in type_:
                    # Past tense verbs only go in past maps
                    type_to_token_map_past[type_].append(token)
                    token2tag_past[token] = type_
                else:
                    # Present tense verbs (v_s/v_p) only go in present maps
                    if "v_s" in type_ or "v_p" in type_:
                        type_to_token_map_present[type_].append(token)
                        token2tag_present[token] = type_
            else:
                # Non-verb tokens (det, n_s, n_p, prep, rel, etc.) go in both
                type_to_token_map_present[type_].append(token)
                type_to_token_map_past[type_].append(token)
                token2tag_past[token] = type_
                token2tag_present[token] = type_

    # Period is mapped to empty string (treated specially)
    token2tag_present["."] = ""
    token2tag_past["."] = ""

    # ========== STEP 2: Load Training Data and Extract Type Sequences ==========
    # Load original training data to understand the distribution of sentence types

    # Present data: PAST input -> PRESENT output (e.g., "dog giggled" -> "dog giggles")
    train_present_from_data = load_tense_inflection_data(
        "train", include_identity=False, include_present=True
    )
    # Past data: PAST input -> PAST output (identity mapping)
    train_past_from_data = load_tense_inflection_data(
        "train", include_identity=True, include_present=False
    )

    # Create functions to convert token sequences to type sequences
    token_seq_to_type_seq_fn_past = lambda token_seq: token_seq_to_type_seq(
        token_seq, token2tag_past
    )
    token_seq_to_type_seq_fn_present = lambda token_seq: token_seq_to_type_seq(
        token_seq, token2tag_present
    )

    # Extract type sequences from past data
    # Example: "the dog giggled" -> "det n_s v_past_intrans"
    past_input_types = (
        train_past_from_data["input"]
        .apply(token_seq_to_type_seq_fn_past)
        .values.tolist()
    )
    past_input_types = np.array(past_input_types)

    # Extract type sequences from present data
    # Input is still PAST tense: "the dog giggled"
    present_input_types = (
        train_present_from_data["input"]
        .apply(token_seq_to_type_seq_fn_past)
        .values.tolist()
    )
    # Output is PRESENT tense: "the dog giggles" -> "det n_s v_s_intrans"
    present_output_types = (
        train_present_from_data["output"]
        .apply(token_seq_to_type_seq_fn_present)
        .values.tolist()
    )

    # Get unique sentence type structures
    unique_past_input_types = np.unique(past_input_types)
    unique_present_input_types = np.unique(present_input_types)


    # ========== STEP 3: Validate Data ==========
    # Sanity checks: ensure past inputs don't have present tense verbs
    for i in range(len(unique_past_input_types)):
        assert "v_p_intrans" not in unique_past_input_types[i]
        assert "v_s_intrans" not in unique_past_input_types[i]
        assert "v_p_trans" not in unique_past_input_types[i]
        assert "v_s_trans" not in unique_past_input_types[i]

    for i in range(len(unique_present_input_types)):
        assert "v_p_intrans" not in unique_past_input_types[i]
        assert "v_s_intrans" not in unique_past_input_types[i]
        assert "v_p_trans" not in unique_past_input_types[i]
        assert "v_s_trans" not in unique_past_input_types[i]

    # Ensure present outputs don't have past tense
    for i in range(len(present_output_types)):
        assert "past" not in present_output_types[i]

    print("total past types:", len(unique_past_input_types))

    # Map each unique present input type to its corresponding output type
    # This captures the transformation pattern (e.g., which verb position needs inflection)
    unique_present_types = {}
    for i in range(len(present_input_types)):
        if present_input_types[i] not in unique_present_types:
            unique_present_types[present_input_types[i]] = present_output_types[i]

    print("total present types:", len(unique_present_types))

    # ========== STEP 4: Generate PAST Tense Samples ==========
    # PAST samples are identity mappings (input == output)
    if not args.present_only:
        # Determine how many past samples to generate
        if not args.keep_present_ratio:
            # Add synthetic past samples on top of original data
            num_past_samples = args.num_samples - len(train_past_from_data)
        else:
            # Maintain the same PAST/PRESENT ratio as original data
            if args.past_ratio is None:
                past_ratio = len(train_past_from_data) / (len(train_present_from_data) + len(train_past_from_data))
            else:
                past_ratio = args.past_ratio  # Use user-specified ratio
            num_past_samples = int(args.num_samples * past_ratio)
            print(f"iso-ratio generation: original past ratio: {past_ratio}, past samples: {num_past_samples}")

        # Start with original training data
        past_pairs = []
        for i in range(len(train_past_from_data['input'])):
            past_pairs.append(f"{train_past_from_data['input'][i]} PAST\t{train_past_from_data['output'][i]}")

        print("used past pairs from training data: ", len(past_pairs))

        # Generate additional synthetic past samples if needed
        # Strategy: Replicate type structures and randomly sample tokens for each type
        if len(past_pairs) < num_past_samples:
            for inp in past_input_types:
                inp_types = inp.split()
                inp_tokens = copy.copy(inp_types)
                # Replace each type with a random token of that type
                for i, type_ in enumerate(inp_types):
                    if type_ in type_to_token_map_past:
                        inp_tokens[i] = np.random.choice(type_to_token_map_past[type_])

                # Identity mapping: input == output
                past_pairs.append(f"{' '.join(inp_tokens)} . PAST\t{' '.join(inp_tokens)} .")

        past_pairs = past_pairs[:num_past_samples]

    # ========== STEP 5: Generate PRESENT Tense Samples ==========
    # PRESENT samples involve tense inflection with subject-verb agreement
    if args.present_only:
        num_present_samples = args.num_samples
    else:
        num_present_samples = args.num_samples - len(past_pairs)

    if args.present_type_ratio is not None:
        # Split sentence types by structural category:
        # - V: Simple verb sentences (e.g., "the dog giggled")
        # - PREP: Prepositional phrase sentences (e.g., "the dog near the cat giggled")
        # - RC: Relative clause sentences (e.g., "the dog that giggled waited")

        unique_input_present_types = {
            'V': [],      # Simple verb constructions
            'PREP': [],   # Prepositional phrase constructions
            'RC': [],     # Relative clause constructions
        }
        unique_output_present_types = {
            'V': [],
            'PREP': [],
            'RC': [],
        }

        # Categorize each sentence type by looking at the 3rd element (index 2)
        # Typical structure: [det, noun, main_element, ...]
        for i in range(len(present_input_types)):
            if "v" in present_input_types[i].split()[2]:
                # Third element is a verb -> simple V construction
                unique_input_present_types['V'].append(present_input_types[i])
                unique_output_present_types['V'].append(present_output_types[i])
            elif present_input_types[i].split()[2] == 'rel':
                # Third element is 'rel' -> relative clause construction
                unique_input_present_types['RC'].append(present_input_types[i])
                unique_output_present_types['RC'].append(present_output_types[i])
            elif present_input_types[i].split()[2] == 'prep':
                # Third element is 'prep' -> prepositional phrase construction
                unique_input_present_types['PREP'].append(present_input_types[i])
                unique_output_present_types['PREP'].append(present_output_types[i])
            else:
                raise ValueError("unknown type")

        # Display original data distribution
        print("original data distribution:")
        print("  len of V: ", len(unique_input_present_types['V']))
        print("  len of PREP: ", len(unique_input_present_types['PREP']))
        print("  len of RC: ", len(unique_input_present_types['RC']))

        # Optionally control the ratio of RC subtypes
        if args.rc_type_ratio is not None:
            print("control RC ratio...")
            # RC can be subject-extracted (RC_sbj) or object-extracted (RC_obj)
            # - RC_sbj (simple): "the dog that giggled" (verb comes right after 'that')
            # - RC_obj (complex): "the dog that the cat saw" (noun comes after 'that')
            rc_type_ratio = [float(x) for x in args.rc_type_ratio.split(",")]
            print("RC ratio: ", rc_type_ratio)
            assert len(rc_type_ratio) == 2
            assert rc_type_ratio[0] + rc_type_ratio[1] == 1.0

            unique_input_present_types["RC_sbj"] = []
            unique_input_present_types["RC_obj"] = []
            unique_output_present_types["RC_sbj"] = []
            unique_output_present_types["RC_obj"] = []

            for i in range(len(unique_input_present_types['RC'])):
                assert unique_input_present_types['RC'][i].split()[2] == 'rel'
                # Determine RC type by looking at element after 'rel' (index 3)
                if unique_input_present_types['RC'][i].split()[3] == 'det':
                    # Object-extracted: "det noun rel det ..." (complex)
                    unique_input_present_types["RC_obj"].append(unique_input_present_types['RC'][i])
                    unique_output_present_types["RC_obj"].append(unique_output_present_types['RC'][i])
                else:
                    # Subject-extracted: "det noun rel verb ..." (simple)
                    assert "v" in unique_input_present_types['RC'][i].split()[3]
                    unique_input_present_types["RC_sbj"].append(unique_input_present_types['RC'][i])
                    unique_output_present_types["RC_sbj"].append(unique_output_present_types['RC'][i])
            # Display original RC subtype distribution
            print("subject RC: ", len(unique_input_present_types['RC_sbj']))
            print("object RC: ", len(unique_input_present_types['RC_obj']))
            quit()

            # Calculate desired number of each RC subtype based on specified ratio
            total_rc_samples = len(unique_input_present_types['RC'])
            num_rc_sbj_samples = int(rc_type_ratio[0] * total_rc_samples)
            num_rc_obj_samples = total_rc_samples - num_rc_sbj_samples

            # Replicate RC subtypes to reach desired counts
            final_rc_sbj_input_samples = []
            final_rc_sbj_output_samples = []
            final_rc_obj_input_samples = []
            final_rc_obj_output_samples = []

            # Repeat subject RC types until we have enough
            while len(final_rc_sbj_input_samples) < num_rc_sbj_samples:
                final_rc_sbj_input_samples += unique_input_present_types['RC_sbj']
                final_rc_sbj_output_samples += unique_output_present_types['RC_sbj']

            # Repeat object RC types until we have enough
            while len(final_rc_obj_input_samples) < num_rc_obj_samples:
                final_rc_obj_input_samples += unique_input_present_types['RC_obj']
                final_rc_obj_output_samples += unique_output_present_types['RC_obj']

            # Trim to exact desired counts
            final_rc_sbj_input_samples = final_rc_sbj_input_samples[:num_rc_sbj_samples]
            final_rc_sbj_output_samples = final_rc_sbj_output_samples[:num_rc_sbj_samples]
            final_rc_obj_input_samples = final_rc_obj_input_samples[:num_rc_obj_samples]
            final_rc_obj_output_samples = final_rc_obj_output_samples[:num_rc_obj_samples]

            # Merge back into RC category
            unique_input_present_types['RC'] = final_rc_sbj_input_samples + final_rc_obj_input_samples
            unique_output_present_types['RC'] = final_rc_sbj_output_samples + final_rc_obj_output_samples
            print("    len of RC_sbj: ", len(final_rc_sbj_input_samples))
            print("    len of RC_obj: ", len(final_rc_obj_input_samples))
            print("    len of RC: ", len(unique_input_present_types['RC']))


        # Calculate number of samples needed for each sentence type
        # Format: "V_ratio,PREP_ratio,RC_ratio" (e.g., "0.5,0.25,0.25")
        present_type_ratio = args.present_type_ratio.split(",")
        present_type_ratio = [float(ratio) for ratio in present_type_ratio]
        assert present_type_ratio[0] + present_type_ratio[1] + present_type_ratio[2] == 1.0
        num_v_samples = int(num_present_samples * present_type_ratio[0])
        num_prep_samples = int(num_present_samples * present_type_ratio[1])
        num_rc_samples = num_present_samples - num_prep_samples - num_v_samples

        # Construct final list of type sequences for each category
        final_present_samples = {
            "V_inp": [],
            "V_out": [],
            "PREP_inp": [],
            "PREP_out": [],
            "RC_inp": [],
            "RC_out": [],
        }

        for present_type in ["V", "PREP", "RC"]:
            if present_type == "V":
                num_type_sampples = num_v_samples
            elif present_type == "PREP":
                num_type_sampples = num_prep_samples
            else:
                num_type_sampples = num_rc_samples

            # If we need fewer samples than available, subsample
            if num_type_sampples < len(unique_input_present_types[present_type]):
                final_present_samples[f'{present_type}_inp'] = unique_input_present_types[present_type][:num_type_sampples]
                final_present_samples[f'{present_type}_out'] = unique_output_present_types[present_type][:num_type_sampples]
            else:
                # If we need more samples than available, replicate types
                final_present_samples[f'{present_type}_inp'] = unique_input_present_types[present_type]
                final_present_samples[f'{present_type}_out'] = unique_output_present_types[present_type]

                # Keep replicating until we have enough
                while len(final_present_samples[f'{present_type}_inp']) < num_type_sampples:
                    final_present_samples[f'{present_type}_inp'] += unique_input_present_types[present_type]
                    final_present_samples[f'{present_type}_out'] += unique_output_present_types[present_type]

                # Trim to exact count
                final_present_samples[f'{present_type}_inp'] = final_present_samples[f'{present_type}_inp'][:num_type_sampples]
                final_present_samples[f'{present_type}_out'] = final_present_samples[f'{present_type}_out'][:num_type_sampples]

        # Display final counts
        print("V samples: ", len(final_present_samples["V_inp"]))
        print("PREP samples: ", len(final_present_samples["PREP_inp"]))
        print("RC samples: ", len(final_present_samples["RC_inp"]))

        # Merge all type sequences into final lists
        final_present_input_types = final_present_samples["V_inp"] + final_present_samples["PREP_inp"] + final_present_samples["RC_inp"]
        final_present_output_types = final_present_samples["V_out"] + final_present_samples["PREP_out"] + final_present_samples["RC_out"]

    # ========== STEP 6: Instantiate Type Sequences with Tokens ==========
    # Convert type sequences to actual token sequences by random sampling
    present_pairs = []
    while len(present_pairs) < num_present_samples:
        for inp, out in zip(final_present_input_types, final_present_output_types):
            # Generate input tokens (past tense)
            inp_types = inp.split()
            inp_tokens = copy.copy(inp_types)
            for i, type_ in enumerate(inp_types):
                if type_ in type_to_token_map_past:
                    inp_tokens[i] = np.random.choice(type_to_token_map_past[type_])

            # Generate output tokens (present tense with agreement)
            out_tokens = copy.copy(inp_tokens)  # Start with same tokens
            out_types = out.split()

            # Apply tense inflection to verbs
            for i, type_ in enumerate(out_types):
                if "v_p" in type_:
                    # Plural verb: use base form (index 0)
                    out_tokens[i] = tense_map[out_tokens[i]][0]
                elif "v_s" in type_:
                    # Singular verb: use -s form (index 1)
                    out_tokens[i] = tense_map[out_tokens[i]][1]

            present_pairs.append(f"{' '.join(inp_tokens)} . PRESENT\t{' '.join(out_tokens)} .")

    print("present pairs: ", len(present_pairs))
    present_pairs = present_pairs[:num_present_samples]

    # ========== STEP 7: Combine and Save ==========
    if args.present_only:
        train_pairs = present_pairs
    else:
        train_pairs = past_pairs + present_pairs
    train_pairs = train_pairs[:args.num_samples]

    # Create output directory if needed
    data_dir = f"{DATA_DIR}/cfg_gen_data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save to file
    output_file = f"{data_dir}/ti_{args.num_types}_types.train"
    with open(output_file, "w") as f:
        f.write("\n".join(train_pairs))
    print("saved data:", output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_types", type=int, default=-1)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--present_only", action="store_true")
    parser.add_argument("--keep_present_ratio", action="store_true", help='flag to control past/present ratio')
    parser.add_argument("--past_ratio", type=float, default=None)
    parser.add_argument("--present_type_ratio", type=str, default=None, help="[prep, rc]")
    parser.add_argument("--rc_type_ratio", type=str, default=None, help="[sbj, obj]")
    args = parser.parse_args()
    main(args)

