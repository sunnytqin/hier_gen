# Data Drives Unstable Hierarchical Generalization in LMs

**Official Repository**

This is the official code repository for the paper:

**"Data Drives Unstable Hierarchical Generalization in LMs"**
*Tian Qin, Naomi Saphra, David Alvarez-Melis*
EMNLP 2025

📄 [Paper](https://aclanthology.org/2025.emnlp-main.593/) | 🔗 [ACL Anthology](https://aclanthology.org/2025.emnlp-main.593/)

## Overview

This repository provides code for training transformer models on question formation and tense inflection tasks to study hierarchical generalization in language models. The code enables investigation of how training data complexity and diversity affect whether models learn hierarchical syntactic rules versus linear n-gram-like shortcuts.

## Setup

### 1. Install Dependencies

```bash
# Create a conda/mamba environment (recommended)
conda create -n hiergenv python=3.11
conda activate hiergenv

# Install the local/forked transformers library first
# (adjust path to your local transformers directory)
pip install -e /path/to/transformers

# Install remaining requirements
pip install -r requirements.txt
```

**Note**: This project requires a local/forked version of the `transformers` library (v4.36.0.dev0). Make sure to install it as an editable package before installing other dependencies.

### 2. Configure WandB (Optional)

If you want to use Weights & Biases logging:

```bash
wandb login
```

Update the `WANDB_ENTITY_NAME` in `train_transformers.py` to your WandB entity name (line 25).

To disable WandB logging, don't specify `--run_name` when running the training script.

## Dataset

The repository includes the question formation dataset:
- `data_utils/question_formation_data/question.train` - 100,000 training examples
- `data_utils/question_formation_data/question.val` - 1,000 validation examples
- `data_utils/question_formation_data/question.test` - 10,000 test examples

## Training

### Quick Start

Run training with default settings from the original experiment:

```bash
# For CPU/local GPU
python train_transformers.py \
  --callback \
  --batch_size 8 \
  --dataset question_qf_disamb \
  --disamb_num 0 \
  --max_train_steps 300000 \
  --encoder_n_layers 6 \
  --lr 1e-4 \
  --num_warmup_steps 10000 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --eval_every 2000 \
  --seed 42 \
  --tied-embedding \
  --run_name qf_disamb_lr1e-4_seed42 \
  --save_every 100000 \
  --save_dir ./checkpoints/ \
  --save_prefix question_qf_disamb_lr1e-4
```

### SLURM Cluster

For SLURM clusters with GPU access:

```bash
sbatch submit_gpu.sh
```

Edit `submit_gpu.sh` to adjust:
- SLURM settings (time, memory, partition, account)
- Training hyperparameters
- Output directory paths

## Model Architecture

The default configuration trains a 6-layer transformer language model with:
- 512-dimensional embeddings
- 8 attention heads
- Tied input/output embeddings
- Standard positional encoding
- Label smoothing: 0.0
- Dropout: 0.1

## Key Arguments

### Dataset Options
- `--dataset`: Dataset name (`question_qf_disamb`, `tense`, etc.)
- `--disamb_num`: Number of disambiguating examples (0 = none)
- `--exclude_identity`: Only include question sentences in training
- `--train_on_decl_only`: Only include declarative sentences

### Model Options
- `--encoder_n_layers`: Number of transformer layers (default: 6)
- `--vec_dim`: Embedding dimension (default: 512)
- `--n_heads`: Number of attention heads (default: 8)
- `--tied-embedding`: Use tied input/output embeddings
- `--gated-model`: Use gated transformer architecture

### Training Options
- `--lr`: Learning rate (default: 1e-4)
- `--batch_size`: Training batch size (default: 8)
- `--max_train_steps`: Maximum training steps (default: 200000)
- `--num_warmup_steps`: Learning rate warmup steps (default: 10000)
- `--weight_decay`: Weight decay coefficient (default: 0.0)
- `--max_grad_norm`: Gradient clipping norm (default: 1.0)

### Evaluation Options
- `--callback`: Enable auxiliary prediction accuracy evaluation
- `--eval_every`: Evaluation frequency in steps (default: 1000)
- `--eval_keys`: Comma-separated eval splits (default: "val,test")

### Checkpointing
- `--save_dir`: Directory to save checkpoints
- `--save_prefix`: Prefix for checkpoint directory names
- `--save_every`: Checkpoint saving frequency (default: 10000)
- `--model_load_path`: Path to load pretrained model

## Output

Training outputs:
- Model checkpoints saved to `{save_dir}/{save_prefix}_{slurm_job_id}/`
- WandB logs (if enabled)
- Training arguments saved as `args.json`

Evaluation metrics:
- Training loss
- Validation/test loss
- Auxiliary prediction accuracy (with `--callback`)

## File Structure

```
hier_gen/
├── train_transformers.py      # Main training script
├── training_utils.py           # Training loop, optimizer, scheduler
├── transformer_helpers.py      # Model creation helpers
├── vocabulary.py               # Vocabulary management
├── collate.py                  # Data batching utilities
├── sequence.py                 # Sequence testing utilities
├── plot.py                     # Plotting utilities
├── util.py                     # General utilities
├── submit_gpu.sh               # SLURM submission script
├── requirements.txt            # Python dependencies
├── data_utils/                 # Dataset loading utilities
│   ├── lm_dataset_helpers.py
│   ├── tense_inflection_helpers.py
│   └── question_formation_data/
├── models/                     # Model architectures
├── interfaces/                 # Model interfaces
├── layers/                     # Neural network layers
└── cfgs/                       # Configuration files
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{qin-etal-2025-data,
    title = "Data Drives Unstable Hierarchical Generalization in {LM}s",
    author = "Qin, Tian  and
      Saphra, Naomi  and
      Alvarez-Melis, David",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.593/",
    doi = "10.18653/v1/2025.emnlp-main.593",
    pages = "11733--11751",
    ISBN = "979-8-89176-332-6",
    abstract = "Early in training, LMs can behave like n-gram models, but eventually, they often learn tree-based syntactic rules and generalize hierarchically out of distribution (OOD). We study this shift using controlled grammar-learning tasks: question formation and tense inflection. We find a model learns to generalize hierarchically if its training data is *complex*{--}in particular, if it includes center-embedded clauses, a special syntactic structure. Under this definition, complex data drives hierarchical rules, while less complex data encourages shortcut learning in the form of n-gram-like linear rules. Furthermore, we find that a model uses rules to generalize, whether hierarchical or linear, if its training data is *diverse*{--}in particular, if it includes many distinct syntax trees in the training set. Under this definition, diverse data promotes stable rule learning, whereas less diverse data promotes memorization of individual syntactic sequences. Finally, intermediate diversity and intermediate complexity form an *unstable regime*, which is characterized by oscillatory learning dynamics and inconsistent behaviors across random seeds. These results highlight the central role of training data in shaping generalization and explain why competing strategies can lead to unstable outcomes."
}
```
