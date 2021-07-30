import argparse


def boolean_string(s):
    if s in {"False", "0", "false"}:
        return False
    if s in {"True", "1", "true"}:
        return False
    raise ValueError("Not a valid boolean string")


def get_parser(
    training: bool, bnb: bool = False, speaker: bool = False
) -> argparse.ArgumentParser:
    """Return an argument parser with the standard VLN-BERT arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["vilbert", "prevalent", "oscar"],
        default="vilbert",
    )

    # fmt: off
    # input/output data handling
    parser.add_argument(
        "--in_memory",
        default=False,
        type=boolean_string,
        help="Store the dataset in memory (default: True)",
    )
    parser.add_argument(
        "--img_feature",
        default="data/matterport-ResNet-101-faster-rcnn-genome.lmdb",
        type=str,
        help="Image features store in an LMDB file",
    )
    parser.add_argument(
        "-j",
        "--num_workers",
        default=3,
        type=int,
        help="Number of workers per gpu (default: 3)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/runs",
        type=str,
        help="The root output directory (default: data/runs)",
    )
    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="The name tag used for saving (default: '')",
    )
    # model configuration
    parser.add_argument(
        "--bert_tokenizer",
        default="bert-base-uncased",
        type=str,
        help="Bert tokenizer model (default: bert-base-uncased)",
    )
    parser.add_argument(
        "--config_file",
        default="data/config/bert_base_6_layer_6_connect.json",
        type=str,
        help="Model configuration file (default: data/config/bert_base_6_layer_6_connect.json)",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Load a pretrained model (default: bert-base-uncased)"
    )
    parser.add_argument(
        "--max_instruction_length",
        default=60,
        type=int,
        help="The maximum number of instruction tokens used by the model (default: 60)",
    )
    parser.add_argument(
        "--max_path_length",
        default=8,
        type=int,
        help="The maximum number of viewpoints tokens used by the model (default: 8)",
    )
    parser.add_argument(
        "--max_num_boxes",
        default=101,
        type=int,
        help="The maximum number of regions used from each viewpoint (default: 101)",
    )
    # Modified
    parser.add_argument(
        "--shortest_path",
        default=True,
        type=boolean_string,
        help="Adding the ground truth trajectory for perturbations (default: True)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        type=str,
        help="Prefix for dataset variants (default: '')",
    )
    parser.add_argument(

        "--beam_prefix",
        default="",
        type=str,
        help="Beam prefix for dataset variants (default: '')",
    )
    parser.add_argument(
        "--highlighted_language",
        action="store_true",
        help="Highlight words in the instruction tokens (default: false)",
    )
    parser.add_argument(
        "--cat_highlight",
        action="store_true",
        help="Concatenate the highlight logit to the vilbert logit (default: false)",
    )
    parser.add_argument(
        "--highlighted_perturbations",
        action="store_true",
        help="Highlight words in the perturbations tokens (default: false)",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="The fixed number of ranked paths to use in inference (default: 30)"
    )
    parser.add_argument(
        "--num_negatives",
        default=2,
        type=int,
        help="The number of negatives per type of negatives (default: 2)"
    )
    parser.add_argument(
        "--shuffler",
        default="different",
        type=str,
        choices=["different", "nonadj", "two"],
        help="Shuffling function (default: different)",
    )
    parser.add_argument(
        "--shuffle_visual_features",
        action='store_true',
        help="Shuffle visual features during training (default: false)",
    )
    parser.add_argument(
        "--perturbation",
        default=False,
        action="store_true",
        help="Using perturbation  (default: False)",
    )
    parser.add_argument(
        "--rank",
        default=-1,
        type=int,
        help="rank for distributed computing on gpus",
    )
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="local_rank for distributed computing on gpus",
    )
    parser.add_argument(
        "--world_size",
        default=-1,
        type=int,
        help="Number of GPUs on which to divide work (default: -1)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="where the run code",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="The size of one batch of training (default: 64)",
    )

    # training configuration
    if training:
        parser.add_argument(
            "--training_mode",
            default='provided',
            choices=['sampled', 'provided', 'augmented'],
            help="The approach to collecting training paths (default: provided)",
        )
        parser.add_argument(
            "--masked_vision",
            action="store_true",
            help="Mask image regions during training (default: false)",
        )
        parser.add_argument(
            "--masked_language",
            action="store_true",
            help="Mask instruction tokens during training (default: false)",
        )
        # parser.add_argument(
        #     "--skip_val",
        #     action="store_true",
        #     help="Skip validation",
        # )
        parser.add_argument(
            "--no_scheduler",
            action="store_true",
            help="Deactivate the scheduler",
        )
        parser.add_argument(
            "--no_ranking",
            action='store_true',
            help="Do not rank trajectories during training (default: false)",
        )
        parser.add_argument(
            "--num_epochs",
            default=20,
            type=int,
            help="Total number of training epochs (default: 20)",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            default=8,
            type=int,
            help="Number of step before a backward pass (default: 8)",
        )
        parser.add_argument(
            "--learning_rate",
            default=4e-5,
            type=float,
            help="The initial learning rate (default: 4e-5)",
        )
        parser.add_argument(
            "--warmup_proportion",
            default=0.2,
            type=float,
            help="Percentage of training to perform a linear lr warmup (default: 0.2)",
        )
        parser.add_argument(
            "--cooldown_factor",
            default=2.0,
            type=float,
            help="Multiplicative factor applied to the learning rate cooldown slope (default: 2.0)",
        )
        parser.add_argument(
            "--weight_decay",
            default=1e-2,
            type=float,
            help="The weight decay (default: 1e-2)"
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Train on a small subset of the dataset (default: false)"
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=1,
            help="Seed the random number generator for training"
        )
        # Modified
        parser.add_argument(
            "--ground_truth_trajectory",
            default=False,
            type=boolean_string,
            help="Adding the ground truth trajectory in the pool of candidate (default: false)",
        )
        parser.add_argument(
            "--hard_mining",
            default=False,
            type=boolean_string,
            help="Applying hard mining during training (default: False)",
        )
        parser.add_argument(
            "--num_beams_train",
            default=4,
            type=int,
            help="The fixed number of ranked paths to use during training (default: 4)"
        )

    if bnb:
        parser.add_argument(
            "--min_path_length",
            default=4,
            type=int,
            help="The fixed number of ranked paths to use in inference (default: 30)"
        )
        parser.add_argument(
            "--min_captioned",
            default=2,
            type=int,
            help="The minimum number of captioned images (default: 7)"
        )
        parser.add_argument(
            "--max_captioned",
            default=7,
            type=int,
            help="The maximum number of captioned images (default: 7)"
        )
        parser.add_argument(
            "--precomputed",
            default="",
            type=str,
            help="Precomputed path for generating better captions (concatenation if '')"
        )
        parser.add_argument(
            "--skeleton",
            default="",
            type=str,
            help="Skeleton path for generating better captions (concatenation if '')"
        )
        parser.add_argument(
            "--combine_dataset",
            default=False,
            action="store_true",
            help="Combine a precomputed dataset with a non precompute dataset (default: False)",
        )
        parser.add_argument(
            "--out_listing",
            default=False,
            action="store_true",
            help="Using photo ids from other listings (default: False)",
        )
        parser.add_argument(
            "--separators",
            default=False,
            action="store_true",
            help="Using multiple separators when joining captions (default: False)",
        )
        parser.add_argument(
            "--bnb_feature",
            default=[
                "data/airbnb-butd-indoor-parts_8-id_0.lmdb",
                "data/airbnb-butd-indoor-parts_8-id_1.lmdb",
                "data/airbnb-butd-indoor-parts_8-id_2.lmdb",
                "data/airbnb-butd-indoor-parts_8-id_3.lmdb",
                "data/airbnb-butd-indoor-parts_8-id_4.lmdb",
                "data/airbnb-butd-indoor-parts_8-id_5.lmdb",
                "data/airbnb-butd-indoor-parts_8-id_6.lmdb",
                "data/airbnb-butd-indoor-parts_8-id_7.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_0.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_1.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_2.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_3.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_4.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_5.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_6.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_7.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_8.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_9.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_10.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_11.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_12.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_13.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_14.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_15.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_16.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_17.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_18.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_19.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_20.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_21.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_22.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_23.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_24.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_25.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_26.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_27.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_28.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_29.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_30.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_31.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_32.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_33.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_34.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_35.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_36.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_37.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_38.lmdb",
#                "data/airbnb-ext-butd-indoor-parts_40-id_39.lmdb",
            ],
            type=str,
            nargs="+",
            help="Image features store in an LMDB file",
        )


    if speaker:
        parser.add_argument(
            "--dataset",
            default="r2r",
            type=str,
            help="Type of dataset",
        )
        parser.add_argument(
            "--np",
            default=False,
            action="store_true",
            help="Add noun phrases before tokens",
        )
        parser.add_argument(
            "--window",
            default=20,
            type=int,
            help="Length for splitting a sentence",
        )
    # fmt: on

    return parser
