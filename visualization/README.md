# Generating Sentence Type Tags

This directory contains scripts to generate sentence type tag files (`.type` files) for question formation data.

## Overview

The `.type` files tag each sentence with its grammatical type structure by converting word tokens to their corresponding grammatical type tags (e.g., "the cat" → "d_sg n_sg").

## Dependencies

The script requires:
- Python 3.6+
- pandas
- numpy (optional, only needed for analysis scripts)

Install pandas if not already available:
```bash
pip install pandas
```

Or add it to `hier_gen/requirements.txt`:
```bash
echo "pandas==2.1.0" >> ../requirements.txt
pip install pandas
```

## Usage

### Basic Usage

To generate type tags for the test split:
```bash
cd hier_gen/visualization
python generate_sentence_type_tags.py --split test
```

This will create: `hier_gen/data_utils/question_formation_data/question.test.type`

### Generate for Different Splits

For training data:
```bash
python generate_sentence_type_tags.py --split train
```

For validation data:
```bash
python generate_sentence_type_tags.py --split val
```

### Custom Filenames

If you have custom data files like `question_D20.train`, use the `--filename` argument:
```bash
python generate_sentence_type_tags.py --split train --filename question_D20
```

This will create: `hier_gen/data_utils/question_formation_data/question_D20.train.type`

## Output Format

The generated `.type` files have the following format:
```
<input_types> <task_type>\t<output_types>
```

Example:
```
d_sg n_sg aux_pres v_p_intrans d_sg n_sg quest	aux_pres d_sg n_sg v_p_intrans d_sg n_sg
d_sg n_sg aux_pres v_p_intrans d_sg n_sg decl	d_sg n_sg aux_pres v_p_intrans d_sg n_sg
```

Where:
- `<input_types>`: Grammatical types of the input sentence
- `<task_type>`: Either "quest" (question) or "decl" (declarative)
- `<output_types>`: Grammatical types of the output sentence

## File Structure

```
hier_gen/
├── visualization/
│   ├── generate_sentence_type_tags.py  # Main script to generate .type files
│   └── README.md                        # This file
├── generate_type_tags.py                # Helper functions
├── cfgs/
│   └── tag_token_map.txt               # Token-to-type mapping
└── data_utils/
    └── question_formation_data/
        ├── question.train
        ├── question.val
        ├── question.test
        └── question.test.type          # Generated output
```

## How It Works

1. **Load token-to-tag mapping**: Reads `cfgs/tag_token_map.txt` to know which tokens map to which grammatical types
2. **Load question formation data**: Reads the input data file (e.g., `question.test`)
3. **Convert tokens to types**: Replaces each word token with its grammatical type
4. **Save type file**: Writes the type sequences to a `.type` file

## Helper Functions

The `generate_type_tags.py` module in the parent directory contains:
- `load_question_formation_data()`: Load question formation data files
- `token_seq_to_type_seq()`: Convert token sequences to type sequences
- `load_token_to_tag_mapping()`: Load the token→type mapping
- `determine_qtype()`: Determine if a transformation is linear or hierarchical

## Troubleshooting

### ModuleNotFoundError: No module named 'pandas'
Make sure pandas is installed in your Python environment:
```bash
pip install pandas
```

### FileNotFoundError
Make sure you're running the script from the `hier_gen/visualization/` directory:
```bash
cd hier_gen/visualization
python generate_sentence_type_tags.py --split test
```

### Input file not found
Verify that the input data file exists:
```bash
ls ../data_utils/question_formation_data/question.test
```

## Examples

### Example 1: Generate all splits
```bash
cd hier_gen/visualization

# Generate for all splits
python generate_sentence_type_tags.py --split train
python generate_sentence_type_tags.py --split val
python generate_sentence_type_tags.py --split test
```

### Example 2: Generate for custom dataset
If you have `question_D150.train`:
```bash
python generate_sentence_type_tags.py --split train --filename question_D150
```
