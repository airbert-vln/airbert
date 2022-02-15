import logging
from typing import List, Tuple, Dict
import os
import random
from pathlib import Path
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

try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    print("Can't load apex...")
    from torch.nn.parallel import DistributedDataParallel as DDP

from tensorboardX import SummaryWriter

from transformers import BertTokenizer

from vilbert.optimization import AdamW, WarmupLinearSchedule
from airbert import Airbert, BERT_CONFIG_FACTORY

from utils.cli import get_parser
from utils.distributed import set_cuda, wrap_distributed_model, get_local_rank
from utils.misc import set_seed, get_output_dir
from utils.dataset import PanoFeaturesReader
from utils.dataset.common import pad_packed
from utils.dataset.beam_dataset import BeamDataset
from utils.dataset.hard_mining import HardMiningDataset
from utils.dataset.perturbate_dataset import PerturbateDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def get_score(data_loader: DataLoader, model: nn.Module, default_gpu: bool):
    if isinstance(data_loader.dataset, PerturbateDataset) and isinstance(data_loader.dataset.dataset, HardMiningDataset):
        dataset: HardMiningDataset = data_loader.dataset.dataset
    elif isinstance(data_loader.dataset, HardMiningDataset):
        dataset = data_loader.dataset
    else:
        raise ValueError(f"Unexpected dataset type ({type(data_loader.dataset)})")

    device = next(model.parameters()).device
    is_training = dataset._training

    batch: Tuple[torch.Tensor, ...]
    for batch in data_loader:
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        output = model(*get_model_input(batch))
        dataset.post_step(output, list(batch))

    dataset._training = is_training

    if default_gpu and hasattr(dataset, "save"):
        dataset.save()


def compute_metrics(batch: List[torch.Tensor], outputs: Dict[str, torch.Tensor], args) -> Tuple[torch.Tensor, Dict[str, float]]:
    # B, num_cand
    opt_mask = get_mask_options(batch)

    local_rank = get_local_rank(args)
    batch_size = get_batch_size(batch)
    device = opt_mask.device

    # calculate the masked vision loss
    vision_loss = torch.tensor(0, device=device)
    if args.masked_vision:
        # num_image_regions, im_feature
        vision_predictions = outputs["vision"]
        vision_predictions = vision_predictions.view(-1, vision_predictions.shape[2])

        vision_target, vision_target_mask = get_vision_target(batch)

        vision_loss = F.kl_div(
            F.log_softmax(vision_predictions, dim=-1),
            vision_target,
            reduction="none",
        )

        vision_loss *= vision_target_mask.unsqueeze(-1).float()
        numel = max(1, torch.sum(vision_target_mask).item())
        vision_loss = torch.sum(vision_loss) / numel

    # calculate the masked language loss
    linguistic_loss = torch.tensor(0, device=device)
    if args.masked_language:
        # (num_words * num_cand * B), voc_size
        voc_size = outputs["language"].shape[-1]
        linguistic_predictions = outputs["language"].view(-1, voc_size )

        linguistic_target = get_linguistic_target(batch)

        linguistic_loss = F.cross_entropy(
            linguistic_predictions, linguistic_target, ignore_index=-1
        )

    # calculate the trajectory re-ranking loss
    ranking_loss = torch.tensor(0, device=device)
    correct = torch.tensor(0, device=device)
    if not args.no_ranking:
        target = get_target(batch)
        vil_logit = pad_packed(outputs["action"].squeeze(1), opt_mask)
        ranking_loss = F.cross_entropy(vil_logit, target, ignore_index=-1)
        # calculate accuracy
        correct = torch.sum(torch.argmax(vil_logit, 1) == target).float()

    # calculate the final loss
    loss = ranking_loss + vision_loss + linguistic_loss
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    # calculate accumulated stats
    reduced_vision_loss = vision_loss.detach().float()
    reduced_linguistic_loss = linguistic_loss.detach().float()
    reduced_ranking_loss = ranking_loss.detach().float()
    reduced_loss = loss.detach().float() * args.gradient_accumulation_steps
    reduced_correct = correct.detach().float()
    reduced_batch_size = torch.tensor(batch_size, device=device).float()

    # TODO: skip this `all_reduce` to speed-up runtime
    if local_rank != -1:
        world_size = float(dist.get_world_size())
        reduced_vision_loss /= world_size
        reduced_linguistic_loss /= world_size
        reduced_ranking_loss /= world_size
        reduced_loss /= world_size
        dist.all_reduce(reduced_vision_loss, op=dist.ReduceOp.SUM) # type: ignore
        dist.all_reduce(reduced_linguistic_loss, op=dist.ReduceOp.SUM) # type: ignore
        dist.all_reduce(reduced_ranking_loss, op=dist.ReduceOp.SUM) # type: ignore
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM) # type: ignore
        dist.all_reduce(reduced_correct, op=dist.ReduceOp.SUM) # type: ignore
        dist.all_reduce(reduced_batch_size, op=dist.ReduceOp.SUM) # type: ignore

    reduced_metrics = {
        "loss/train": reduced_loss,
        "loss/ranking": reduced_ranking_loss,
        "loss/vision": reduced_loss,
        "loss/linguistic": reduced_vision_loss,
        "accuracy/train": reduced_correct / reduced_batch_size,
    }

    # if default_gpu and args.debug:
    #     logger.info(
    #         f"[train] step: {step + 1} "
    #         f"vision loss: {reduced_vision_loss:0.2f} "
    #         f"language loss: {reduced_linguistic_loss:0.2f} "
    #         f"ranking loss: {reduced_ranking_loss:0.2f} "
    #         f"loss: {reduced_loss:0.2f} "
    #         f"accuracy: {reduced_correct / reduced_batch_size:0.2f} "
    #         f"lr: {scheduler.get_lr()[0]:0.1e}"
    #     )

    return loss, reduced_metrics


def main():
    # ----- #
    # setup #
    # ----- #

    # command line parsing
    parser = get_parser(training=True)
    args = parser.parse_args()

    # validate command line arguments
    if not (args.masked_vision or args.masked_language) and args.no_ranking:
        parser.error(
            "No training objective selected, add --masked_vision, "
            "--masked_language, or remove --no_ranking"
        )

    set_seed(args)

    # get device settings
    default_gpu, rank, device = set_cuda(args)

    # create output directory
    save_folder = get_output_dir(args)
    if default_gpu:
        save_folder.parent.mkdir(exist_ok=True, parents=True)

        print(args)

    # ------------ #
    # data loaders #
    # ------------ #

    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    features_reader = PanoFeaturesReader(args.img_feature)
    train_params = {}
    vln_path = f"data/task/{args.prefix}R2R_train.json"

    if args.training_mode == "provided":
        train_params["beam_path"] = f"data/beamsearch/{args.beam_prefix}beams_train.json"
        if default_gpu:
            logger.info("using provided training trajectories")
            logger.info(f"VLN path: {vln_path}")
            logger.info(f"Beam path: {train_params['beam_path']}")

        if args.hard_mining:
            TrainDataset = HardMiningDataset
            train_params["save_folder"] = str(save_folder)
        else:
            TrainDataset = BeamDataset
    else:
        raise ValueError(f"Unknown training_mode for {args.training_mode}")

    if default_gpu:
        logger.info("Loading train dataset")

    train_dataset: Dataset = TrainDataset(
        **train_params,
        vln_path=vln_path,
        tokenizer=tokenizer,
        features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_path_length=args.max_path_length,
        max_num_boxes=args.max_num_boxes,
        num_beams=args.num_beams_train,
        num_beams_strict=False,
        training=True,
        masked_vision=args.masked_vision,
        masked_language=args.masked_language,
        default_gpu=default_gpu,
        ground_truth_trajectory=args.ground_truth_trajectory,
        highlighted_language=args.highlighted_language,
        shuffle_visual_features=args.shuffle_visual_features,
        num_negatives=args.num_negatives,
        shuffler=args.shuffler,
    )

    #TODO add perturbation inside BeamDataset?
    if args.perturbation:
        if not isinstance(train_dataset, BeamDataset):
            raise ValueError(f"Unexpected dataset type ({type(train_dataset)})")
        train_dataset = PerturbateDataset(
            dataset=train_dataset,
            num_negatives=args.num_negatives,
            perturbate_path=f"data/task/{args.prefix}R2R_train.json",
            shortest_path=args.shortest_path,
            highlighted_perturbations=args.highlighted_perturbations,
        )

    val_seen_params = {}
    val_unseen_params = {}

    TestDataset = BeamDataset

    if default_gpu:
        logger.info("Loading val datasets")

    val_seen_dataset = TestDataset(
        **val_seen_params,
        vln_path=f"data/task/{args.prefix}R2R_val_seen.json",
        beam_path=f"data/beamsearch/{args.beam_prefix}beams_val_seen.json",
        tokenizer=tokenizer,
        features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_path_length=args.max_path_length,
        max_num_boxes=args.max_num_boxes,
        num_beams=args.num_beams,
        num_negatives=args.num_negatives,
        num_beams_strict=False,
        training=False,
        masked_vision=False,
        masked_language=False,
        default_gpu=default_gpu,
        ground_truth_trajectory=False,
        highlighted_language=args.highlighted_language,
        shuffle_visual_features=False,
    )

    val_unseen_dataset = TestDataset(
        **val_unseen_params,
        vln_path=f"data/task/{args.prefix}R2R_val_unseen.json",
        beam_path=f"data/beamsearch/{args.beam_prefix}beams_val_unseen.json",
        tokenizer=tokenizer,
        features_reader=features_reader,
        max_instruction_length=args.max_instruction_length,
        max_path_length=args.max_path_length,
        max_num_boxes=args.max_num_boxes,
        num_beams=args.num_beams,
        num_beams_strict=False,
        training=False,
        masked_vision=False,
        masked_language=False,
        default_gpu=default_gpu,
        ground_truth_trajectory=False,
        highlighted_language=args.highlighted_language,
        shuffle_visual_features=False,
        num_negatives=args.num_negatives,
    )

    # in debug mode only run on a subset of the datasets
    if args.debug:
        train_dataset = Subset(
            train_dataset,
            np.random.choice(range(len(train_dataset)), size=128, replace=False), # type: ignore
        )
        val_seen_dataset = Subset(
            val_seen_dataset,
            np.random.choice(range(len(val_seen_dataset)), size=64, replace=False), # type: ignore
        )
        val_unseen_dataset = Subset(
            val_unseen_dataset,
            np.random.choice(range(len(val_unseen_dataset)), size=64, replace=False), # type: ignore
        )

    local_rank = get_local_rank(args)

    if local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        val_seen_sampler = SequentialSampler(val_seen_dataset)
        val_unseen_sampler = SequentialSampler(val_unseen_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        val_seen_sampler = DistributedSampler(val_seen_dataset)
        val_unseen_sampler = DistributedSampler(val_unseen_dataset)

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

    config = BERT_CONFIG_FACTORY[args.model_name].from_json_file(args.config_file)
    config.cat_highlight = args.cat_highlight # type: ignore
    config.no_ranking = args.no_ranking # type: ignore
    config.masked_language = args.masked_language # type: ignore
    config.masked_vision = args.masked_vision # type: ignore
    config.model_name = args.model_name
    start_epoch = 0

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
        WarmupLinearSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            t_total=adjusted_t_total,
            last_epoch=-1,
        )
        if not args.no_scheduler
        else MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0) # type: ignore
    )

    # load checkpoint of the optimizer
    weights_path = Path(args.from_pretrained)
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu")
        if 'optimizer_state_dict' in state_dict:
            state_dict = state_dict["optimizer_state_dict"]
            optimizer.load_state_dict(state_dict)
        if 'scheduler_state_dict' in state_dict:
            state_dict = state_dict["scheduler_state_dict"]
            scheduler.load_state_dict(state_dict)
        if "epoch" in state_dict:
            start_epoch = state_dict["epoch"]


    # --------------- #
    # before training #
    # --------------- #

    # save the parameters
    if default_gpu:
        with open(save_folder /  "config.txt", "w") as fid:
            print(f"{datetime.now()}", file=fid)
            print("\n", file=fid)
            print(vars(args), file=fid)
            print("\n", file=fid)
            print(config, file=fid)

    # loggers
    if default_gpu:
        writer = SummaryWriter(
            logdir=save_folder / "logging", flush_secs=30
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
    for epoch in range(start_epoch, args.num_epochs):
        if default_gpu and args.debug:
            logger.info(f"epoch {epoch}")

        # Modified:
        if isinstance(train_data_loader.sampler, DistributedSampler):
            train_data_loader.sampler.set_epoch(epoch) 

        if args.hard_mining:
            if default_gpu:
                logger.info("setting the beam scores")
            get_score(train_data_loader, model, default_gpu) # type: ignore
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
            logger.info("saving the model")

        # save the model every epoch
        model_path = save_folder / f"pytorch_model_{epoch + 1}.bin"
        if default_gpu:
            model_state = (
                    model.module.state_dict() # type: ignore
                if hasattr(model, "module")
                else model.state_dict()
            )
            torch.save(model_state, model_path)

        if default_gpu and args.debug:
            logger.info(f"running validation")

        # run validation
        if not args.no_ranking:
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
                    best_seen_path = save_folder / "pytorch_model_best_seen.bin"
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
                    best_unseen_path = save_folder / "pytorch_model_best_unseen.bin"
                    shutil.copyfile(model_path, best_unseen_path)

    # -------------- #
    # after training #
    # -------------- #

    if default_gpu:
        writer.close()


def train_epoch(
    epoch, model, optimizer, scheduler, data_loader, writer, default_gpu, args
) -> None:
    device = next(model.parameters()).device

    model.train()
    model.zero_grad()

    for step, batch in enumerate(tqdm(data_loader, disable= not (default_gpu))):
        # load batch on gpu
        batch = tuple(
            t.cuda(device=device, non_blocking=True) if hasattr(t, "cuda") else t
            for t in batch
        )

        # get the model output
        outputs = model(*get_model_input(batch))
        
        loss, reduced_metrics = compute_metrics(batch, outputs, args) # type: ignore

        # backward pass
        loss.backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        # write stats to tensorboard
        if default_gpu:
            global_step = step + epoch * len(data_loader)
            reduced_metrics["learning_rate/train"] = float(scheduler.get_lr()[0])
            for name, reduced_loss in reduced_metrics.items():
                writer.add_scalar(name, reduced_loss, global_step=global_step)


def val_epoch(epoch: int, model, tag, data_loader, writer, default_gpu, args, global_step):
    device = next(model.parameters()).device
    local_rank = get_local_rank(args)

    # validation
    model.eval()
    stats = torch.zeros(3, device=device).float()
    for step, batch in enumerate(data_loader):
        # load batch on gpu
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        batch_size = get_batch_size(batch)
        target = get_target(batch)

        # get the model output
        inputs = get_model_input(batch)
        outputs = model(*inputs)
        opt_mask = batch[13]

        if not args.no_ranking:
            vil_logit = pad_packed(outputs["action"].squeeze(1), opt_mask)

            # calculate loss
            loss = F.binary_cross_entropy_with_logits(vil_logit, target.float())

            # calculate success rate of the top scoring beam
            correct = torch.sum(
                target.gather(1, torch.argmax(vil_logit, 1).view(-1, 1))
            ).float()

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

    if local_rank != -1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM) # type: ignore

    # write stats to tensorboard
    if default_gpu:
        writer.add_scalar(
            f"loss/bce_{tag}", stats[0] / stats[2], global_step=global_step
        )
        writer.add_scalar(
            f"accuracy/sr_{tag}", stats[1] / stats[2], global_step=global_step
        )

    return stats[1] / stats[2]


# ------------- #
# batch parsing #
# ------------- #

# batch format:
# 0:target, 1:image_features, 2:image_locations, 3:image_mask, 4:image_targets,
# 5:image_targets_mask, 6:instr_tokens, 7:instr_mask, 8:instr_targets, 9:instr_highlights, 10:segment_ids,
# 11:co_attention_mask, 12:item_id


def get_model_input(batch):
    (
        _,
        image_features,
        image_locations,
        image_mask,
        _,
        _,
        instr_tokens,
        instr_mask,
        _,
        instr_highlights,
        segment_ids,
        co_attention_mask,
        _,
        opt_mask,
    ) = batch

    # remove padding samples
    image_features = image_features[opt_mask]
    image_locations = image_locations[opt_mask]
    image_mask = image_mask[opt_mask]
    instr_tokens = instr_tokens[opt_mask]
    instr_mask = instr_mask[opt_mask]
    instr_highlights = instr_highlights[opt_mask]
    segment_ids = segment_ids[opt_mask]

    # transform batch shape
    co_attention_mask = co_attention_mask.view(
        -1, co_attention_mask.size(2), co_attention_mask.size(3)
    )

    return (
        instr_tokens,
        image_features,
        image_locations,
        segment_ids,
        instr_mask,
        image_mask,
        co_attention_mask,
        instr_highlights,
    )


def get_batch_size(batch):
    return batch[1].size(0)



def get_target(batch):
    return batch[0]

def get_mask_options(batch) -> torch.Tensor:
    return batch[13]

def get_num_options(batch):
    mask = get_mask_options(batch)
    return batch[6][mask].size(0)


def get_linguistic_target(batch):
    opt_mask = get_mask_options(batch)
    return batch[8][opt_mask].flatten()


def get_highlights(batch):
    opt_mask = get_mask_options(batch)
    return batch[9][opt_mask]


def get_vision_target(batch):
    opt_mask = get_mask_options(batch)
    return (
        batch[4][opt_mask].flatten(0, 1),
        batch[5][opt_mask].flatten(),
    )



if __name__ == "__main__":
    main()
