# File: dvc_audio.py
# Original Author(s): VidChapters Team (https://github.com/antoyang/VidChapters)
# Modified by: Daniel Vousden
#
# Description:
# This file extends the VidChapters dense video captioning training and evaluation pipeline to support audio features.
# It introduces audio input handling via Wav2Vec 2.0, integrates it into the multimodal input structure alongside video and ASR features,
# and modifies both training and generation logic to accommodate this additional modality.
#
# Key Modifications:
# - Added support for loading and batching audio features.
# - Updated model training loop to include audio modality input.
# - Extended evaluation pipeline to pass audio features to the model.
# - Maintained compatibility with the existing VidChapters structure and evaluation scripts (e.g., dvc_eval, SODA).
#
# License: MIT 


import os
os.environ["APEX_FUSED_LAYER_NORM"] = "0"
os.environ["TRANSFORMERS_NO_APEX"] = "1"

import torch
import numpy as np
import random
import json
import math
import sys
from typing import Iterable
import argparse
import time
import datetime
import re
import logging
from util import dist
from torch.utils.data import DataLoader, DistributedSampler
from collections import namedtuple
from functools import reduce
# Added dataset using audio.
from dvc_dataset_audio import build_densevideocaptioning_dataset_audio as build_densevideocaptioning_dataset
from dvc_dataset_audio import densevideocaptioning_collate_fn_audio as densevideocaptioning_collate_fn
from model import build_vid2seq_model, _get_tokenizer
from args import get_args_parser
from util.misc import adjust_learning_rate
from util.metrics import MetricLogger
from dvc_eval import eval_dvc, eval_soda

def train_one_epoch(model, data_loader, optimizer, device, epoch, args):
    logger = logging.getLogger(__name__)
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    num_training_steps = int(len(data_loader) * args.epochs)

    for i_batch, batch_dict in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        video = batch_dict["video"].to(device)
        output_tokens = batch_dict["output_tokens"].to(device)
        # Adds audio
        if args.use_audio_features and "audio" in batch_dict:
            audio_features = batch_dict["audio"].to(device)
        else:
            audio_features = None

        if "input_tokens" in batch_dict:
            input_tokens = batch_dict["input_tokens"].to(device)
            input_tokenized = {"input_ids": input_tokens, "attention_mask": input_tokens != 0}
        else:
            input_tokenized = {"input_ids": None, "attention_mask": None}

        output_tokenized = {"input_ids": output_tokens, "attention_mask": output_tokens != 0}
        # Passes to the model
        loss_dict, _ = model(video=video, input_tokenized=input_tokenized, output_tokenized=output_tokenized, audio=audio_features)
        loss = loss_dict["loss"]

        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_reduced = sum(loss_dict_reduced.values())
        loss_value = loss_reduced.item()

        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training.")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        adjust_learning_rate(optimizer, curr_step=epoch * len(data_loader) + i_batch, num_training_steps=num_training_steps, args=args)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats after epoch {epoch}: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, data_loader, device, args, split="test", dataset_name="youcook"):
    logger = logging.getLogger(__name__)
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"{split}:"

    res = {}
    for i_batch, batch_dict in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        duration = batch_dict["duration"]
        video = batch_dict["video"].to(device)

        # Handle audio features if available
        if args.use_audio_features and "audio" in batch_dict:
            audio_features = batch_dict["audio"].to(device)
        else:
            audio_features = None

        if "input_tokens" in batch_dict:
            input_tokens = batch_dict["input_tokens"].to(device)
            input_tokenized = {"input_ids": input_tokens, "attention_mask": input_tokens != 0}
        else:
            input_tokenized = {"input_ids": None, "attention_mask": None}

        # Generate predictions, added audio
        output = model.generate(
            video=video,
            input_tokenized=input_tokenized,
            audio=audio_features,
            use_nucleus_sampling=args.num_beams == 0,
            num_beams=args.num_beams,
            max_length=args.max_output_tokens,
            min_length=1,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            num_captions=1,
            temperature=1
        )

        # Process predictions and timestamps
        for i, vid in enumerate(batch_dict["video_id"]):
            sequences = re.split(r'(?<!<)\s+(?!>)', output[i])
            indexes = [j for j in range(len(sequences) - 1) if sequences[j].startswith('<time=') and sequences[j + 1].startswith('<time=')]
            last_processed = -2
            res[vid] = []

            for j, idx in enumerate(indexes):
                if idx == last_processed + 1:
                    continue
                seq = [sequences[k] for k in range(idx + 2, indexes[j + 1] if j < len(indexes) - 1 else len(sequences)) if sequences[k] != '<time=']
                if not seq:
                    continue
                text = ' '.join(seq)
                start_token = int(re.search(r'\<time\=(\d+)\>', sequences[idx]).group(1))
                end_token = int(re.search(r'\<time\=(\d+)\>', sequences[idx + 1]).group(1))
                start = float(start_token) * float(duration[i]) / float(args.num_bins - 1)
                end = float(end_token) * float(duration[i]) / float(args.num_bins - 1)
                if end <= start:
                    continue
                res[vid].append({"sentence": text, "timestamp": [start, end]})
                last_processed = idx

    # Save predictions to a file
    pred_path = os.path.join(args.save_dir, f"{dataset_name}_{split}_preds.json")
    json.dump({'results': res}, open(pred_path, "w"))

    # Evaluate with dvc_eval and soda_eval
    references = [args.youcook_val_json_path] if dataset_name == "youcook" else []
    metrics = eval_dvc(pred_path, references, tious=[0.3, 0.5, 0.7, 0.9], max_proposals_per_video=1000)

    # Also evaluate with eval_soda for additional metrics
    soda_metrics = eval_soda(pred_path, references, verbose=False)

    # Combine the metrics
    metrics.update(soda_metrics)

    # Print each metric like in dvc.py
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    dist.init_distributed_mode(args)

    if dist.is_main_process() and args.save_dir and not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)

    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tokenizer = _get_tokenizer(args.model_name, args.num_bins)

    dataset_train = build_densevideocaptioning_dataset("youcook", "train", args, tokenizer)
    dataset_val = build_densevideocaptioning_dataset("youcook", "val", args, tokenizer)

    sampler_train = DistributedSampler(dataset_train) if args.distributed else torch.utils.data.RandomSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dataset_val)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler_train, collate_fn=densevideocaptioning_collate_fn, num_workers=args.num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size_val, sampler=sampler_val, collate_fn=densevideocaptioning_collate_fn, num_workers=args.num_workers)

    model = build_vid2seq_model(args, tokenizer)
    model.to(device)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    if args.load:
        logger.info(f"loading from {args.load}")
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.resume and not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1

    if not args.eval:
        logger.info("Start training")
        start_time = time.time()

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)

            train_one_epoch(model, dataloader_train, optimizer, device, epoch, args)
            val_res = evaluate(model, dataloader_val, device, args, split="val", dataset_name="youcook")

            if dist.is_main_process() and args.save_dir:
                checkpoint_path = os.path.join(args.save_dir, f"continued_checkpoint_epoch_{epoch}.pth")
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "args": args}, checkpoint_path)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training time {total_time_str}")

    else:
        logger.info("Start evaluation")
        results = evaluate(model, dataloader_val, device, args, split="val", dataset_name="youcook")
        if dist.is_main_process() and args.save_dir:
            json.dump(results, open(os.path.join(args.save_dir, "val_results.json"), "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
   
    main(args)
