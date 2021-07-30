"""
Train a speaker model on R2R
"""
import logging
from typing import List, Tuple, Dict
import copy
import os
import random
import shutil
import sys
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from apex.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, BertTokenizer
from vilbert.optimization import AdamW, WarmupLinearSchedule
from vilbert.vilbert import BertConfig
from vln_bert import VLNBert
from utils.cli import get_parser
from utils.dataset import PanoFeaturesReader
from utils.dataset.speak_dataset import SpeakDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

Batch = Dict[str, torch.Tensor]

def main():
    # ----- #
    # setup #
    # ----- #

    # command line parsing
    parser = get_parser(training=True, speaker=True)
    args = parser.parse_args()
    # FIXME how to do it properly in bash?
    args.perturbations = [p for pert in args.perturbations for p in pert.split(" ")]

    # validate command line arguments
    if not (args.masked_vision or args.masked_language) and args.no_ranking:
        parser.error(
            "No training objective selected, add --masked_vision, "
            "--masked_language, or remove --no_ranking"
        )

    # set seed
    if args.seed:
        seed = args.seed
        if args.local_rank != -1:
            seed += args.local_rank
        torch.manual_seed(seed)
        np.random.seed(seed) # type: ignore
        random.seed(seed)

    # get device settings
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of synchronizing
        # nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
        n_gpu = 1

    # check if this is the default gpu
    default_gpu = True
    if args.local_rank != -1 and dist.get_rank() != 0:
        default_gpu = False
    if default_gpu:
        logger.info(f"Playing with {n_gpu} GPUs")

    # create output directory
    save_folder = os.path.join(args.output_dir, f"run-{args.save_name}")
    if default_gpu and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # ------------ #
    # data loaders #
    # ------------ #

    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)
    if not isinstance(tokenizer, BertTokenizer):
        raise ValueError("fix mypy")
    features_reader = PanoFeaturesReader(args.img_feature)
    vln_path = f"data/task/{args.prefix}R2R_train.json"

    if default_gpu:
        logger.info("using provided training trajectories")
        logger.info(f"VLN path: {vln_path}")

    if default_gpu:
        logger.info("Loading train dataset")

    train_dataset: Dataset = SpeakDataset(
        vln_path=vln_path,
        skeleton_path="np_train.json" if args.np else "",
        tokenizer=tokenizer,
        features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_path_length=args.max_path_length,
        max_num_boxes=args.max_num_boxes,
        default_gpu=default_gpu,
    )

    if default_gpu:
        logger.info("Loading val datasets")

    val_seen_dataset = SpeakDataset(
        vln_path=f"data/task/{args.prefix}R2R_val_seen.json",
        skeleton_path="np_val_seen.json" if args.np else "",
        tokenizer=tokenizer,
        features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_path_length=args.max_path_length,
        max_num_boxes=args.max_num_boxes,
        default_gpu=default_gpu,
    )

    val_unseen_dataset = SpeakDataset(
        vln_path=f"data/task/{args.prefix}R2R_val_unseen.json",
        skeleton_path="np_val_unseen.json" if args.np else "",
        tokenizer=tokenizer,
        features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_path_length=args.max_path_length,
        max_num_boxes=args.max_num_boxes,
        default_gpu=default_gpu,
    )

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        val_seen_sampler = SequentialSampler(val_seen_dataset)
        val_unseen_sampler = SequentialSampler(val_unseen_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        val_seen_sampler = DistributedSampler(val_seen_dataset)
        val_unseen_sampler = DistributedSampler(val_unseen_dataset)

    # adjust the batch size for distributed training
    batch_size = args.batch_size // args.gradient_accumulation_steps
    if args.local_rank != -1:
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
    val_seen_data_loader = DataLoader(
        val_seen_dataset,
        sampler=val_seen_sampler,
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_unseen_data_loader = DataLoader(
        val_unseen_dataset,
        sampler=val_unseen_sampler,
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

    config = BertConfig.from_json_file(args.config_file)
    config.cat_highlight = args.cat_highlight # type: ignore
    config.convert_mask = True # type: ignore

    if len(args.from_pretrained) == 0:  # hack for catching --from_pretrained ""
        model = VLNBert(config)
    else:
        model = VLNBert.from_pretrained(
            args.from_pretrained, config, default_gpu=default_gpu
        )

    if default_gpu:
        logger.info(
            f"number of parameters: {sum(p.numel() for p in model.parameters())}"
        )

    # move/distribute model to device
    model.to(device)
    if args.local_rank != -1:
        model = DDP(model, delay_allreduce=True)
        if default_gpu:
            logger.info("using distributed data parallel")
    # elif n_gpu > 1:
    #     model = torch.nn.DataParallel(model) # type: ignore
    #     if default_gpu:
    #         logger.info("using data parallel")

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
        WarmupLinearSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            t_total=adjusted_t_total,
            last_epoch=-1,
        )
        if not args.no_scheduler
        else MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0) # type: ignore
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
            log_dir=os.path.join(save_folder, "logging"), flush_secs=30
        )
    else:
        writer = None

    # -------- #
    # training #
    # -------- #

    # run training
    if default_gpu:
        logger.info("starting training...")

    best_seen_success_rate, best_unseen_success_rate = 0, 0
    for epoch in range(args.num_epochs):
        if default_gpu and args.debug:
            logger.info(f"epoch {epoch}")

        if args.local_rank > -1:
            train_data_loader.sampler.set_epoch(epoch) # type: ignore

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
        model_path = os.path.join(save_folder, f"pytorch_model_{epoch + 1}.bin")
        if default_gpu:
            model_state = (
                    model.module.state_dict() # type: ignore
                if hasattr(model, "module")
                else model.state_dict()
            )
            torch.save(
                {
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                },
                model_path
            )

        if default_gpu and args.debug:
            logger.info(f"running validation")

        # run validation
        global_step = (epoch + 1) * len(train_data_loader)

        # run validation on the "val seen" split
        with torch.no_grad():
            seen_success_rate = val_epoch(
                epoch,
                model,
                "val_seen",
                val_seen_data_loader,
                writer,
                default_gpu,
                args,
                global_step,
            )
            if default_gpu:
                logger.info(
                    f"[val_seen] epoch: {epoch + 1} success_rate: {seen_success_rate.item():.3f}"
                )

        # save the model that performs the best on val seen
        if seen_success_rate > best_seen_success_rate:
            best_seen_success_rate = seen_success_rate
            if default_gpu:
                best_seen_path = os.path.join(
                    save_folder, "pytorch_model_best_seen.bin"
                )
                shutil.copyfile(model_path, best_seen_path) # type: ignore

        # run validation on the "val unseen" split
        with torch.no_grad():
            unseen_success_rate = val_epoch(
                epoch,
                model,
                "val_unseen",
                val_unseen_data_loader,
                writer,
                default_gpu,
                args,
                global_step,
            )
            if default_gpu:
                logger.info(
                    f"[val_unseen] epoch: {epoch + 1} success_rate: {unseen_success_rate.item():.3f}"
                )

        # save the model that performs the best on val unseen
        if unseen_success_rate > best_unseen_success_rate:
            best_unseen_success_rate = unseen_success_rate
            if default_gpu:
                best_unseen_path = os.path.join(
                    save_folder, "pytorch_model_best_unseen.bin"
                )
                shutil.copyfile(model_path, best_unseen_path)

    # -------------- #
    # after training #
    # -------------- #

    if default_gpu:
        writer.close()


def rollout(batch: Batch, model: nn.Module, window: int
        ) :
    """
    we split the batch over sequences of $window tokens.
    This reduces the burden on memory usage.
    """
    # get the model input and output
    instruction_length = batch["target_tokens"].shape[1]
    batch_size = get_batch_size(batch)
    device = get_device(batch)
    inputs = get_model_input(batch)
    # import ipdb
    # ipdb.set_trace()
    # B, N
    target = get_target(batch) # inputs["instr_tokens"][:, 0]
    # B, N, N
    pred_mask = get_mask_predictions(batch)
    # B, N
    pad_or_sep = (batch["target_tokens"] == 102) | (batch["target_tokens"] == 0)
    pad_or_sep = pad_or_sep.squeeze(1)

    map_loss = torch.tensor(0.).to(device)
    map_correct = torch.tensor(0.).to(device)
    map_batch_size  = torch.tensor(0.).to(device)

    for start in range(0, instruction_length, window):
        small_inputs = {
                key: tensor[:, start: start+ window].flatten(0, 1) for key, tensor in inputs.items()
        }
        small_target = target[:, start+1:start+window+1].flatten()

        output = model(**small_inputs)

        # N * W * B
        small_mask = pred_mask[:, start : start + window].flatten()
        # N * W * B x V
        predictions = output[2].view(-1, output[2].shape[-1])
        # W * B x V
        predictions = predictions[small_mask]
        # W x B
        instr = predictions.argmax(1).view(batch_size, -1)

        # calculate the final loss on non-padding tokens 
        loss = F.cross_entropy(predictions, small_target, ignore_index=0)

        # backward pass
        if model.training:
            loss.backward()

        # calculate accuracy
        # remove pad tokens and sep tokens
        small_pad = pad_or_sep[0,start+1: start+window+1 ].flatten()
        correct = torch.sum(instr.flatten()[small_pad] == small_target[small_pad]).detach().float()

        # calculate accumulated stats
        map_batch_size += batch_size
        map_loss += loss.detach().float()
        map_correct += correct.detach().float()

    map_loss = torch.true_divide(map_loss.sum(), map_batch_size) # type: ignore
    map_correct = torch.true_divide(map_correct.sum(), map_batch_size) # type: ignore

    return map_batch_size.float(), map_loss.float(), map_correct.float()

def train_epoch(
    epoch, model, optimizer, scheduler, data_loader, writer, default_gpu, args
) -> None:
    device = next(model.parameters()).device
    model.train()

    batch: Batch
    for step, batch in enumerate(tqdm(data_loader, disable=False)): # not (default_gpu))):
        if step < 78:
            continue
        # load batch on gpu
        batch = {
                k: t.cuda(device=device, non_blocking=True) if hasattr(t, "cuda") else t
            for k, t in batch.items()
        }

        batch_size, loss, correct = rollout(batch, model, args.window)
        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps
            correct /= args.gradient_accumulation_steps


        # write stats to tensorboard
        if default_gpu:
            global_step = step + epoch * len(data_loader)
            writer.add_scalar("loss/train", loss.float(), global_step=global_step)
            writer.add_scalar(
                "accuracy/train",
                correct.float(),
                global_step=global_step,
            )
            writer.add_scalar(
                "learning_rate/train", scheduler.get_lr()[0], global_step=global_step
            )


        if args.local_rank != -1:
            world_size = float(dist.get_world_size())
            loss /= world_size
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_size, op=dist.ReduceOp.SUM)

        if default_gpu and args.debug:
            logger.info(
                f"[train] step: {step + 1} "
                f"loss: {loss:0.2f} "
                f"accuracy: {correct / batch_size:0.2f} "
                f"lr: {scheduler.get_lr()[0]:0.1e}"
            )
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()


def val_epoch(epoch: int, model, tag, data_loader, writer, default_gpu, args, global_step):
    device = next(model.parameters()).device

    # validation
    model.eval()
    stats = torch.zeros(3, device=device).float()
    for step, batch in enumerate(data_loader):
        # load batch on gpu
        batch = {
                k: t.cuda(device=device, non_blocking=True) if hasattr(t, "cuda") else t
            for k, t in batch.items()
        }

        # get the model output
        batch_size, loss, correct = rollout(batch, model, args.window)

        # accumulate
        stats[0] += loss
        stats[1] += correct
        stats[2] += batch_size

        if default_gpu and args.debug:
            logger.info(
                f"[{tag}] step: {step + 1} "
                f"running loss: {stats[0] / stats[2]:0.2f} "
                f"running success rate: {stats[1] / stats[2]:0.2f}"
            )

    if args.local_rank != -1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    # write stats to tensorboard
    if default_gpu:
        writer.add_scalar(
            f"loss/vce_{tag}", stats[0] / stats[2], global_step=global_step
        )
        writer.add_scalar(
            f"accuracy/sr_{tag}", stats[1] / stats[2], global_step=global_step
        )

    return stats[1] / stats[2]


# ------------- #
# batch parsing #
# ------------- #

# batch format:
# 1:image_features, 2:image_locations, 3:image_mask,
# 5:image_targets_mask, 6:instr_tokens, 7:instr_mask, 8:instr_targets, 9:instr_highlights, 10:segment_ids,
# 11:co_attention_mask, 12:item_id

def get_instr_length(batch: Batch):
    return batch["instr_tokens"].shape[1]

def get_instr_mask(batch: Batch) -> torch.Tensor:
    return batch["instr_mask"].squeeze(1)

def get_model_input(batch: Batch) -> Dict[str, torch.Tensor]:
    batch_size = get_batch_size(batch)
    num_tokens = get_instr_length(batch)

    # duplicate for each word token
    image_features = batch["image_features"].unsqueeze(1).repeat(1, num_tokens - 1, 1, 1) 
    image_locations = batch["image_boxes"].unsqueeze(1).repeat(1, num_tokens - 1,  1, 1)
    image_mask = batch["image_masks"].unsqueeze(1).repeat(1, num_tokens - 1, 1) 
    instr_tokens = batch["instr_tokens"].unsqueeze(1).repeat(1, num_tokens - 1, 1)
    segment_ids = batch["segment_ids"].unsqueeze(1).repeat(1, num_tokens - 1, 1)
    instr_mask = batch["instr_mask"].unsqueeze(1).repeat(1, num_tokens - 1, 1)

    # create triangular masks
    tri = (
        torch.ones((num_tokens - 1, num_tokens)) 
        .tril(0)
        .bool()
        .repeat(batch_size, 1, 1)
        . transpose(0, 1)
        .reshape(-1, num_tokens)
        .to(instr_mask.device)
    )
    instr_mask = torch.logical_and(instr_mask, tri) # type: ignore

    # transform batch shape
    co_attention_mask = batch["co_attention_mask"].view(
        -1, batch["co_attention_mask"].size(2), batch["co_attention_mask"].size(3)
    )

    return {
            "instr_tokens": instr_tokens,
            "image_features": image_features,
            "image_locations": image_locations,
            "token_type_ids": segment_ids,
            "attention_mask": instr_mask,
            "image_attention_mask": image_mask,
            "co_attention_mask": co_attention_mask,
    }


def get_batch_size(batch: Batch):
    return batch["instr_tokens"].shape[0]


def get_target(batch: Batch) -> torch.Tensor:
    return batch["target_tokens"]

def get_device(batch: Batch):
    return batch["instr_tokens"].device

def get_mask_predictions(batch: Batch) -> torch.Tensor:
    target_length = batch["target_tokens"].shape[1] 
    instruction_length = get_instr_length(batch) - target_length
    batch_size = get_batch_size(batch)
    device = get_device(batch)

    diag = torch.diag(torch.tensor([1] * instruction_length), diagonal=target_length).bool().to(device)
    diag = diag[:-target_length]
    diag[-1] = 0
    diag = diag.repeat(batch_size, 1, 1)
    return diag

if __name__ == "__main__":
    main()
