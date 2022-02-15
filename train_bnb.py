import os
import random
import shutil
import sys
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Subset,
    ConcatDataset,
)
from torch.utils.data.distributed import DistributedSampler

try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    print("Can't load apex...")
    from torch.nn.parallel import DistributedDataParallel as DDP

from tensorboardX import SummaryWriter

# from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertTokenizer

from vilbert.optimization import AdamW, WarmupLinearSchedule, ConstantLRSchedule
from airbert import Airbert, BERT_CONFIG_FACTORY

from utils.cli import get_parser
from utils.dataset import BnBFeaturesReader
from utils.dataset.bnb_dataset import BnBDataset
from utils.dataset.bnb_precomputed_dataset import PrecomputedBnBDataset
from utils.misc import set_seed, get_logger, get_output_dir
from utils.distributed import set_cuda, wrap_distributed_model, get_local_rank
from train import train_epoch, val_epoch, get_score

logger = get_logger(__name__)


def main():
    # ----- #
    # setup #
    # ----- #

    # command line parsing
    parser = get_parser(training=True, bnb=True)
    # TODO clean up this parameter name
    parser.add_argument("--final", default=False, action="store_true")
    args = parser.parse_args()


    # validate command line arguments
    if not (args.masked_vision or args.masked_language) and args.no_ranking:
        parser.error(
            "No training objective selected, add --masked_vision, "
            "--masked_language, or remove --no_ranking"
        )

    # initialize
    set_seed(args)
    default_gpu, rank, device = set_cuda(args)

    # create output directory
    save_folder = get_output_dir(args)
    if default_gpu:
        save_folder.parent.mkdir(exist_ok=True, parents=True)

    # ------------ #
    # data loaders #
    # ------------ #

    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    if not isinstance(tokenizer, BertTokenizer):
        raise ValueError("Fix mypy issue")

    if default_gpu:
        logger.info(f"Loading bnb_feature from {args.bnb_feature}")
    features_reader = BnBFeaturesReader(args.bnb_feature)
    caption_path = f"data/bnb/{args.prefix}bnb_train.json"
    testset_path = f"data/bnb/{args.prefix}testset.json"
    if default_gpu:
        logger.info("using provided training trajectories")
        logger.info(f"Caption path: {caption_path}")
        logger.info(f"Testset path: {testset_path}")
        logger.info("Loading train dataset")

    separators = ("then", "and", ",", ".") if args.separators else ("[SEP]",)

    # TODO: clean up the dataset creation
    if args.final:
        # Using ConcatenateInstructionGenerator
        train_dataset1 = BnBDataset(
            caption_path=caption_path.replace("np+", ""),
            testset_path=testset_path.replace("np+", ""),
            tokenizer=tokenizer,
            skeleton_path="",
            features_reader=features_reader,
            max_instruction_length=args.max_instruction_length,
            max_length=args.max_path_length,
            min_length=args.min_path_length,
            max_captioned=args.max_captioned,
            min_captioned=args.min_captioned,
            max_num_boxes=args.max_num_boxes,
            num_positives=1,
            num_negatives=args.num_negatives,
            masked_vision=args.masked_vision,
            masked_language=args.masked_language,
            training=True,
            shuffler=args.shuffler,
            out_listing=args.out_listing,
            separators=separators,
        )
        # Using RephraseInstructionGenerator
        train_dataset2 = BnBDataset(
            caption_path=caption_path,
            testset_path=testset_path,
            tokenizer=tokenizer,
            skeleton_path=args.skeleton,
            features_reader=features_reader,
            max_instruction_length=args.max_instruction_length,
            max_length=args.max_path_length,
            min_length=args.min_path_length,
            max_captioned=args.max_captioned,
            min_captioned=args.min_captioned,
            max_num_boxes=args.max_num_boxes,
            num_positives=1,
            num_negatives=args.num_negatives,
            masked_vision=args.masked_vision,
            masked_language=args.masked_language,
            training=True,
            shuffler=args.shuffler,
            out_listing=args.out_listing,
            separators=separators,
        )
        train_dataset = ConcatDataset([train_dataset1, train_dataset2])

    elif args.precomputed == "":
        train_dataset = BnBDataset(
            caption_path=caption_path,
            testset_path=testset_path,
            tokenizer=tokenizer,
            skeleton_path=args.skeleton,
            features_reader=features_reader,
            max_instruction_length=args.max_instruction_length,
            max_length=args.max_path_length,
            min_length=args.min_path_length,
            max_captioned=args.max_captioned,
            min_captioned=args.min_captioned,
            max_num_boxes=args.max_num_boxes,
            num_positives=1,
            num_negatives=args.num_negatives,
            masked_vision=args.masked_vision,
            masked_language=args.masked_language,
            training=True,
            shuffler=args.shuffler,
            out_listing=args.out_listing,
            separators=separators,
        )
    elif args.combine_dataset:
        train_dataset1 = BnBDataset(
            caption_path=caption_path,
            testset_path=testset_path,
            tokenizer=tokenizer,
            skeleton_path=args.skeleton,
            features_reader=features_reader,
            max_instruction_length=args.max_instruction_length,
            max_length=args.max_path_length,
            min_length=args.min_path_length,
            max_captioned=args.max_captioned,
            min_captioned=args.min_captioned,
            max_num_boxes=args.max_num_boxes,
            num_positives=1,
            num_negatives=args.num_negatives,
            masked_vision=args.masked_vision,
            masked_language=args.masked_language,
            training=True,
            shuffler=args.shuffler,
            out_listing=args.out_listing,
            separators=separators,
        )
        train_dataset2 = PrecomputedBnBDataset(
            precomputed_path=args.precomputed,
            tokenizer=tokenizer,
            features_reader=features_reader,
            max_instruction_length=args.max_instruction_length,
            max_length=args.max_path_length,
            min_length=args.min_path_length,
            max_captioned=args.max_captioned,
            min_captioned=args.min_captioned,
            max_num_boxes=args.max_num_boxes,
            num_positives=1,
            num_negatives=args.num_negatives,
            masked_vision=args.masked_vision,
            masked_language=args.masked_language,
            training=True,
            shuffler=args.shuffler,
        )
        train_dataset = ConcatDataset([train_dataset1, train_dataset2])
    else:
        train_dataset = PrecomputedBnBDataset(
            precomputed_path=args.precomputed,
            tokenizer=tokenizer,
            features_reader=features_reader,
            max_instruction_length=args.max_instruction_length,
            max_length=args.max_path_length,
            min_length=args.min_path_length,
            max_captioned=args.max_captioned,
            min_captioned=args.min_captioned,
            max_num_boxes=args.max_num_boxes,
            num_positives=1,
            num_negatives=args.num_negatives,
            masked_vision=args.masked_vision,
            masked_language=args.masked_language,
            training=True,
            shuffler=args.shuffler,
        )

    if default_gpu:
        logger.info(f"Dataset length {len(train_dataset)}")
        logger.info("Loading val datasets")

    test_dataset = BnBDataset(
        caption_path=f"data/bnb/{args.prefix}bnb_test.json",
        testset_path=testset_path,
        skeleton_path=args.skeleton,
        tokenizer=tokenizer,
        features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_length=args.max_path_length,
        min_length=args.min_path_length,
        max_captioned=args.max_captioned,
        min_captioned=args.min_captioned,
        max_num_boxes=args.max_num_boxes,
        num_positives=1,
        num_negatives=args.num_negatives,
        masked_vision=False,
        masked_language=False,
        training=False,
        out_listing=args.out_listing,
        separators=separators,
    )

    # in debug mode only run on a subset of the datasets
    if args.debug:
        train_dataset = Subset(
            train_dataset,
            np.random.choice(range(len(train_dataset)), size=128, replace=False),  # type: ignore
        )
        test_dataset = Subset(
            test_dataset,
            np.random.choice(range(len(test_dataset)), size=64, replace=False),  # type: ignore
        )

    local_rank = get_local_rank(args)
    # print("Local rank is", local_rank)

    if local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        test_sampler = SequentialSampler(test_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)

    # adjust the batch size for distributed training
    batch_size = args.batch_size // args.gradient_accumulation_steps
    if local_rank != -1:
        batch_size = batch_size // dist.get_world_size()
    if default_gpu:
        logger.info(f"batch_size: {batch_size}")

    if default_gpu:
        logger.info(f"Creating dataloader")

    # create data loaders
    train_data_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- #
    # model #
    # ----- #
    if default_gpu:
        logger.info(f"Loading model")

    config = BERT_CONFIG_FACTORY[args.model_name].from_json_file(args.config_file)
    config.cat_highlight = args.cat_highlight # type: ignore
    config.no_ranking = args.no_ranking # type: ignore
    config.masked_language = args.masked_language # type: ignore
    config.masked_vision = args.masked_vision # type: ignore
    config.model_name = args.model_name

    model: nn.Module
    if len(args.from_pretrained) == 0:  # hack for catching --from_pretrained ""
        model = Airbert(config)
    else:
        model = Airbert.from_pretrained(
            args.from_pretrained, config, default_gpu=default_gpu
        )

    if default_gpu:
        logger.info(
            f"number of parameters: {sum(p.numel() for p in model.parameters())}"
        )

    # move/distribute model to device
    model.to(device)
    model = wrap_distributed_model(model, local_rank)

    if default_gpu:
        with open(save_folder / "model.txt", "w") as fid:
            fid.write(str(model))

    # ------------ #
    # optimization #
    # ------------ #

    # set parameter specific weight decay
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": 0.0},
        {"params": [], "weight_decay": args.weight_decay},
    ]
    for name, param in model.named_parameters():
        if any(nd in name for nd in no_decay):
            optimizer_grouped_parameters[0]["params"].append(param)
        else:
            optimizer_grouped_parameters[1]["params"].append(param)

    # optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,)

    # calculate learning rate schedule
    t_total = (
        len(train_data_loader) // args.gradient_accumulation_steps
    ) * args.num_epochs
    warmup_steps = args.warmup_proportion * t_total
    adjusted_t_total = warmup_steps + args.cooldown_factor * (t_total - warmup_steps)
    scheduler = (
        ConstantLRSchedule(optimizer)
        if args.no_scheduler
        else WarmupLinearSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            t_total=adjusted_t_total,
            last_epoch=-1,
        )
    )

    # --------------- #
    # before training #
    # --------------- #

    # save the parameters
    if default_gpu:
        with open(os.path.join(save_folder, "config.txt"), "w") as fid:
            print(f"{datetime.now()}", file=fid)
            print("\n", file=fid)
            print(vars(args), file=fid)
            print("\n", file=fid)
            print(config, file=fid)

    # loggers
    if default_gpu:
        writer = SummaryWriter(
            logdir=os.path.join(save_folder, "logging"), flush_secs=30
        )
    else:
        writer = None

    # -------- #
    # training #
    # -------- #

    # run training
    if default_gpu:
        logger.info("starting training...")

    best_success_rate = 0
    for epoch in range(args.num_epochs):
        if default_gpu and args.debug:
            logger.info(f"epoch {epoch}")

        if isinstance(train_data_loader.sampler, DistributedSampler):
            train_data_loader.sampler.set_epoch(epoch)

        if args.hard_mining:
            if default_gpu:
                logger.info("setting the beam scores")
            get_score(train_data_loader, model, default_gpu)
            if default_gpu:
                logger.info("the beam scores are set")

        # train for one epoch
        train_epoch(
            epoch,
            model,
            optimizer,
            scheduler,
            train_data_loader,
            writer,
            default_gpu,
            args,
        )

        if default_gpu and args.debug:
            logger.info(f"saving the model")
        # save the model every epoch
        if default_gpu:
            if hasattr(model, "module") and isinstance(model.module, nn.Module):
                net: nn.Module = model.module
            elif isinstance(model, nn.Module):
                net = model
            else:
                raise ValueError("Can't find the Module here")

            model_path = os.path.join(save_folder, f"pytorch_model_{epoch + 1}.bin")
            torch.save(
                {
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                },
                model_path,
            )
        else:
            model_path = ""

        if default_gpu and args.debug:
            logger.info(f"running validation")
        # run validation
        if not args.no_ranking:
            global_step = (epoch + 1) * len(train_data_loader)

            # run validation on the "val seen" split
            with torch.no_grad():
                success_rate = val_epoch(
                    epoch,
                    model,
                    "test",
                    test_data_loader,
                    writer,
                    default_gpu,
                    args,
                    global_step,
                )
                if default_gpu:
                    logger.info(
                        f"[test] epoch: {epoch + 1} success_rate: {success_rate.item():.3f}"
                    )

            # save the model that performs the best on val seen
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                if default_gpu:
                    best_test_path = os.path.join(
                        save_folder, "pytorch_model_best_test.bin"
                    )
                    shutil.copyfile(model_path, best_test_path)

    # -------------- #
    # after training #
    # -------------- #

    if default_gpu:
        writer.close()


if __name__ == "__main__":
    main()
