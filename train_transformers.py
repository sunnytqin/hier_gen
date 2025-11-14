import numpy as np
import random
import os
import torch
import json
from training_utils import *

import argparse
from data_utils import (
    build_datasets_lm,
    build_datasets_enc_dec,
    build_datasets_tense_inflection,
    build_datasets_ti_enc_dec,
    build_datasets_lm_cls,
)
from data_utils.tense_inflection_helpers import (
    build_datasets_ti_cls,
    build_datasets_ti_mlm,
)
from transformer_helpers import *
import torch.nn.functional as F
from data_utils.lm_dataset_helpers import (
    eval_lm_callback,
    eval_cls_callback,
    eval_lm_sent_prob_callback,
)
from data_utils.tense_inflection_helpers import (
    eval_callback_tense_inflection,
    eval_ti_cls_callback,
    eval_ti_mlm_callback,
)

WANDB_ENTITY_NAME = "harvardml"


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


### Change this for your own system as appropriate
def working_dir():
    dir_name = os.getcwd()
    return dir_name


def get_base_transformer_model(
    args, in_vocab, out_vocab, num_roles=None, model_name=None
):

    model = create_model(
        len(in_vocab),
        len(out_vocab),
        args.vec_dim,
        args.n_heads,
        args.encoder_n_layers,
        args.decoder_n_layers,
        mode=args.mode,
        tied_embedding=args.tied_embedding,
        dropout=args.dropout,
    )
    if model_name:
        print("loading pretrained model from {}".format(model_name))
        if ".pickle" in model_name:
            model.load_state_dict(
                torch.load(model_name, map_location=torch.device("cpu"))
                )
        else:
            model.load_state_dict(
                torch.load(model_name, map_location=torch.device("cpu"))['model_state_dict']
                )
    interface = create_model_interface(model, label_smoothing=args.label_smoothing)
    return model, interface


def get_base_transformer_lm(args, in_vocab, model_name=None):
    try:
        model = create_lm(
            len(in_vocab),
            args.vec_dim,
            args.n_heads,
            args.encoder_n_layers,
            mode=args.mode,
            use_pos_embeddig=not args.no_pos_enc,
            pos_scale=args.pos_scale,
            gated_model=args.gated_model,
            dropout=args.dropout,
            tied_embedding=args.tied_embedding,
        )
    except AttributeError:
        model = create_lm(
            len(in_vocab),
            args.vec_dim,
            args.n_heads,
            args.encoder_n_layers,
            tied_embedding=args.tied_embedding,
        )
    if model_name:
        print("loading pretrained model from {}".format(model_name))
        if ".pickle" in model_name:
            model.load_state_dict(
                torch.load(model_name, map_location=torch.device("cpu"))
                )
        else:
            loaded_state_dict = torch.load(model_name, map_location=torch.device("cpu"))['model_state_dict']
            adjusted_state_dict = {key.replace('model.', ''): value for key, value in loaded_state_dict.items()}
            model.load_state_dict(adjusted_state_dict)

    try:
        interface = create_model_interface(
            model,
            is_lm=True,
            is_null_encoder=(args.mode != "enc_dec"),
            label_smoothing=args.label_smoothing,
            is_prefix_lm=args.is_prefix_lm,
        )
    except AttributeError:
        interface = create_model_interface(
            model,
            is_lm=True,
            is_null_encoder=False,
            # label_smoothing=args.label_smoothing,
        )
    return model, interface


def get_base_transformer_cls(args, in_vocab, out_vocab, model_name=None):
    try:
        model = create_cls(
            len(in_vocab),
            len(out_vocab),
            args.vec_dim,
            args.n_heads,
            args.encoder_n_layers,
            use_pos_embeddig=not args.no_pos_enc,
            causal_encoder=args.causal_encoder,
        )
    except AttributeError:
        model = create_cls(
            len(in_vocab),
            len(out_vocab),
            args.vec_dim,
            args.n_heads,
            args.encoder_n_layers,
            causal_encoder=args.causal_encoder,
        )
    if model_name:
        print("loading pretrained model from {}".format(model_name))
        model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))

    interface = create_model_interface(model, is_cls=True, is_lm=False)
    return model, interface


def main_lm(args):
    print("args: ", args)
    out_vocab = None

    ############################# get dataset #############################
    if args.dataset == "tense" or "tense" in args.dataset:
        if args.mode != "enc":
            if not args.not_lm:
                datasets, in_vocab, _ = build_datasets_tense_inflection(
                    data_name=args.dataset,
                    include_only_present=args.exclude_identity,
                    include_past_simple=args.include_past_simple,
                    include_past_simple_verb=args.include_past_simple_verb,
                    include_past_complex=args.include_past_complex,
                    include_past_complex_verb=args.include_past_complex_verb,
                    include_present_simple=args.include_present_simple,
                    include_present_simple_verb=args.include_present_simple_verb,
                    include_present_complex=args.include_present_complex,
                    include_present_complex_verb=args.include_present_complex_verb,
                    include_subject_rc_obj=args.include_subject_rc_obj,
                    include_subject_rc_subj=args.include_subject_rc_subj,
                    splits=["train", "val", "test", "val_rc", "val_prep", "test_rc", "test_prep", "val_rc_sbj", "val_rc_obj", "test_rc_sbj", "test_rc_obj"],
                )
            else:
                datasets, in_vocab, in_sents, out_sents = build_datasets_ti_enc_dec(
                    include_only_present=args.exclude_identity,
                    include_only_past_and_simple_present=args.pretrain,
                )
        else:
            (
                datasets,
                in_vocab,
                out_vocab,
                in_sentences,
                out_mvs,
            ) = build_datasets_ti_cls()
    elif args.dataset == "qf_disamb":
        if args.disamb_num == 0:
            filename = "question"
        else:
            filename = (
                "question_disamb"
                if args.disamb_num == -1
                else f"question_disamb_{args.disamb_num}"
            )
        datasets, in_vocab, in_sentences = build_datasets_lm(
            filename_prefix=filename,
            include_only_quest=args.exclude_identity, # quest only
            include_only_decls=args.train_on_decl_only, # del only
            include_only_decls_nd_simpl_ques=args.pretrain, # not used
            include_only_complex_sents=args.train_on_compl_only, # both decl and quest, complex only
            include_only_simple_sents=args.train_on_simple_only,# both decl and quest, simple only
            include_only_hier_sents=args.train_on_hier_only, # use only hier decl, all quest
            include_only_linear_sents=args.train_on_linear_only, # use only linear decl, all quest
            exclude_complex_decls=args.exclude_complex_decls, # all quest and simple decl (depth=1)
            exclude_simple_decls=args.exclude_simple_decls, # all quest and complex decl (depth=2)
            exclude_middle_decls=args.exclude_middle_decls, # all quest and complex decl (depth=3)
            exclude_complex_quest=args.exclude_complex_quest, # all decl and simple quest
            exclude_simple_quest=args.exclude_simple_quest, # all decl and complex quest
            use_specified_decls=args.use_specified_decls, # specify multiple decl types, all quest
            splits=['train', 'val', 'test', 'test_obj', 'test_sbj']
        )
        if args.subset_train_frac < 1.0:
            subselect_size = int(len(datasets['train']) * args.subset_train_frac)
            datasets['train'] = torch.utils.data.Subset(datasets['train'], range(subselect_size))
        print("train set: ", len(datasets['train']))
    elif "question_D" in args.dataset or "question_S" in args.dataset:
        datasets, in_vocab, in_sentences = build_datasets_lm(
            filename_prefix=args.dataset, test_filename_prefix="question",
            include_only_quest=args.exclude_identity, # quest only
            include_only_decls=args.train_on_decl_only, # del only
            include_only_decls_nd_simpl_ques=args.pretrain, # not used
            include_only_complex_sents=args.train_on_compl_only, # both decl and quest, complex only
            include_only_simple_sents=args.train_on_simple_only,# both decl and quest, simple only
            include_only_hier_sents=args.train_on_hier_only, # use only hier quest, no decl
            include_only_linear_sents=args.train_on_linear_only, # use only linear quest, no decl
            exclude_complex_decls=args.exclude_complex_decls, # all quest and simple decl (depth=1)
            exclude_simple_decls=args.exclude_simple_decls, # all quest and complex decl (depth=2)
            exclude_middle_decls=args.exclude_middle_decls, # all quest and complex decl (depth=3)
            exclude_complex_quest=args.exclude_complex_quest, # all decl and simple quest
            exclude_simple_quest=args.exclude_simple_quest, # all decl and complex quest
            use_specified_decls=args.use_specified_decls, # specify multiple decl types, all quest
            splits=['train', 'val', 'test']
        )
        if args.subset_train_frac < 1.0:
            subselect_size = int(len(datasets['train']) * args.subset_train_frac)
            datasets['train'] = torch.utils.data.Subset(datasets['train'], range(subselect_size))
        print("train set: ", len(datasets['train']))

    else:
        if args.mode != "enc":
            if not args.not_lm:
                datasets, in_vocab, _ = build_datasets_lm(
                    include_only_quest=args.exclude_identity,
                    include_only_decls_nd_simpl_ques=args.pretrain,
                    include_only_complex_sents=args.train_on_compl_only,
                )
            else:
                datasets, in_vocab, in_sents, out_sents = build_datasets_enc_dec(
                    include_only_quest=args.exclude_identity,
                    include_only_decls_nd_simpl_ques=args.pretrain,
                )
        else:
            (
                datasets,
                in_vocab,
                out_vocab,
                in_sentences,
                out_auxs,
            ) = build_datasets_lm_cls()

    ############################ getting model ############################
    if args.mode != "enc":
        if not args.not_lm:
            print("getting lm model")
            model, interface = get_base_transformer_lm(
                args, in_vocab, model_name=args.model_load_path
            )
        else:
            model, interface = get_base_transformer_model(
                args, in_vocab, in_vocab, model_name=args.model_load_path
            )
    else:
        assert out_vocab is not None
        model, interface = get_base_transformer_cls(
            args, in_vocab, out_vocab, model_name=args.model_load_path
        )

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        device = "cpu"
    model.to(device)
    print("LM model:", model)
    # print model params
    for i, (name, param) in enumerate(model.named_parameters()):
        print(name, param[1:10])
        if i > 5:
            break

    # print("LM interface:", interface)
    # count number of trainable params
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}".format(num_params))

    if args.save_dir is not None:
        dir_path = working_dir()
        args.save_dir = os.path.join(dir_path, args.save_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    ############################ getting callback ############################
    if args.callback:
        if (
            args.dataset in ["lm", "qf_disamb", "qf_disamb_order", "qf", "qf_w_qid"]
            or ("question_D" in args.dataset) or ("question_" in args.dataset)
        ):
            if args.mode != "enc":
                if not args.not_lm:
                    callback_fn = {
                        "aux": lambda split: eval_lm_callback(
                            model, in_vocab, split, is_prefix_lm=args.is_prefix_lm
                        ),
                        # "sent_prob": lambda split: eval_lm_sent_prob_callback(
                        #     interface, in_vocab, split
                        # ),
                    }
                else:
                    callback_fn = lambda split: eval_lm_callback(model, in_vocab, split)
            else:
                assert out_vocab is not None
                callback_fn = lambda split: eval_cls_callback(
                    model, in_vocab, out_vocab, split
                )

        elif args.dataset == "tense" or "tense" in args.dataset:
            if args.mode != "enc":
                callback_fn = {
                    "aux": lambda split: eval_callback_tense_inflection(
                    model, in_vocab, split, args
                )
                }
            else:
                assert out_vocab is not None
                callback_fn = {
                    "aux": lambda split: eval_ti_cls_callback(
                    model, in_vocab, out_vocab, split
                )
                }
        else:
            raise Exception
    else:
        callback_fn = None

    if args.eval_keys == "":
        eval_keys = ["val", "test"]
    else:
        eval_keys = args.eval_keys.split(",")

    ######################### train loop #########################
    train_loop(
        args,
        interface,
        datasets["train"],
        {key: datasets[key] for key in eval_keys},
        # {"test": datasets["test"]},
        device,
        args.save_dir,
        in_vocab=in_vocab,
        callback_fn=callback_fn,
        eval_every=args.eval_every,
        max_steps=args.max_train_steps,
        train_batch_size=args.batch_size,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model architecture
    parser.add_argument("--vec_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ff_multiplier", default=4, type=int, help="Feed-forward dimension multiplier")
    parser.add_argument("--encoder_n_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--decoder_n_layers", type=int, default=2, help="Number of decoder layers")
    parser.add_argument("--mode", type=str, default="enc_dec", choices=["enc_dec", "enc"], help="Model mode")
    parser.add_argument("--not_lm", action="store_true", help="Use encoder-decoder model instead of LM")
    parser.add_argument("--tied-embedding", action="store_true", help="Use tied input/output embeddings")
    parser.add_argument("--gated-model", action="store_true", help="Use gated transformer model")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--no-pos-enc", action="store_true", help="Disable positional encoding")
    parser.add_argument("--pos_scale", type=float, default=1.0, help="Positional encoding scale")
    parser.add_argument("--causal_encoder", action="store_true", help="Use causal encoder (for mode=enc)")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing coefficient")

    # Dataset and data loading
    parser.add_argument("--dataset", type=str, default="lm", help="Dataset name (qf_disamb, question_D*, tense)")
    parser.add_argument("--subset_train_frac", type=float, default=1.0, help="Fraction of training data to use")

    # Question formation dataset options
    parser.add_argument("--disamb_num", type=int, default=-1, help="Number of disambiguating examples (0=none)")
    parser.add_argument("--exclude_identity", action="store_true", help="Only include question sentences")
    parser.add_argument("--train_on_decl_only", action="store_true", help="Only include declarative sentences")
    parser.add_argument("--train_on_compl_only", action="store_true", help="Only include complex sentences")
    parser.add_argument("--train_on_simple_only", action='store_true', help="Only include simple sentences")
    parser.add_argument("--train_on_hier_only", action="store_true", help="Only include hierarchical sentences")
    parser.add_argument("--train_on_linear_only", action="store_true", help="Only include linear sentences")
    parser.add_argument("--exclude_complex_decls", action="store_true", help="Exclude complex declaratives (depth>1)")
    parser.add_argument("--exclude_simple_decls", action="store_true", help="Exclude simple declaratives (depth=1)")
    parser.add_argument("--exclude_middle_decls", action="store_true", help="Exclude middle depth declaratives")
    parser.add_argument("--exclude_complex_quest", action="store_true", help="Exclude complex questions")
    parser.add_argument("--exclude_simple_quest", action="store_true", help="Exclude simple questions")
    parser.add_argument("--use_specified_decls", type=str, default=None, help="Specify decl aux counts, e.g., '1,3'")

    # Tense inflection dataset options
    parser.add_argument("--include_past_simple", action="store_true", help="Include past simple tense")
    parser.add_argument("--include_past_simple_verb", action="store_true", help="Include past simple with verb")
    parser.add_argument("--include_past_complex", action="store_true", help="Include past complex tense")
    parser.add_argument("--include_past_complex_verb", action="store_true", help="Include past complex with verb")
    parser.add_argument("--include_present_simple", action="store_true", help="Include present simple tense")
    parser.add_argument("--include_present_simple_verb", action="store_true", help="Include present simple with verb")
    parser.add_argument("--include_present_complex", action="store_true", help="Include present complex tense")
    parser.add_argument("--include_present_complex_verb", action="store_true", help="Include present complex with verb")
    parser.add_argument("--include_subject_rc_obj", action="store_true", help="Include subject RC with object")
    parser.add_argument("--include_subject_rc_subj", action="store_true", help="Include subject RC with subject")
    parser.add_argument("--pretrain", action="store_true", help="Pretrain mode")

    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay coefficient")
    parser.add_argument("--continue_lr", action="store_true", help="Continue with saved learning rate")
    parser.add_argument("--num_warmup_steps", type=int, default=10000, help="LR warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--no_decay_lr", action="store_true", help="Disable learning rate decay")
    parser.add_argument("--max_train_steps", type=int, default=200000, help="Maximum training steps")

    # Evaluation and checkpointing
    parser.add_argument("--callback", action="store_true", help="Enable evaluation callbacks")
    parser.add_argument("--eval_keys", type=str, default="", help="Comma-separated eval split names")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluation frequency (steps)")
    parser.add_argument("--save_every", type=int, default=10000, help="Checkpoint save frequency (steps)")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--save_prefix", type=str, default=None, help="Prefix for checkpoint directory")
    parser.add_argument("--model_load_path", type=str, default="", help="Path to load pretrained model")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--is_prefix_lm", action="store_true", help="Use prefix LM mode")

    # Logging
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name (if None, logging disabled)")

    args = parser.parse_args()
    set_seed(args)

    if args.run_name is None: mode = 'disabled'
    else: mode = 'online'
    if "qf_disamb" in args.dataset or "question_D" in args.dataset or "question_" in args.dataset:
        wandb_logger = wandb.init(
            project="hier-genv", entity=WANDB_ENTITY_NAME, config=vars(args),
            name=args.run_name,
            mode=mode
        )
    # elif "passiv" in args.dataset:
    #     wandb_logger = wandb.init(
    #         project="structural-grokking-passiv", entity=WANDB_ENTITY_NAME, config=vars(args),
    #         name=args.run_name,
    #         mode=mode
    #     )
    elif "tense" in args.dataset:
        wandb_logger = wandb.init(
            project="hier-genv-tense", entity=WANDB_ENTITY_NAME, config=vars(args),
            name=args.run_name,
            mode=mode
        )
    else:
        raise Exception("Invalid dataset")

    # To work with wandb sweeps
    args = AttrDict((wandb_logger.config))

    # if args.save_prefix is not None:
    #     args.save_dir = os.path.join(args.save_dir,
    #                                  f"{args.save_prefix}-encL{args.encoder_n_layers}-decL{args.decoder_n_layers}-LR{args.lr}-Nheads{args.n_heads}-EmbSize{args.vec_dim}-TiedEmb{args.tied_embedding}-Seq2Seq{args.not_lm}-Mode{args.mode}-PrefixLM{args.is_prefix_lm}-{args.seed}/")

    slurm_id = os.environ.get("SLURM_JOB_ID")
    if args.save_dir is not None:
        if slurm_id is not None:
            args.save_dir = os.path.join(args.save_dir, f"{args.save_prefix}_{slurm_id}/")

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    if slurm_id is not None:
        wandb.run.name = "{}-{}".format(args.run_name, slurm_id)

    wandb.run.save()

    main_lm(args)
