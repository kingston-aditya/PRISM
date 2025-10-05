import argparse
import logging
import PIL
import numpy as np
import datasets
from datasets import load_dataset, Image
from collections import OrderedDict
from matplotlib import pyplot as plt

import itertools

from pathlib import Path
from absl import app
import io
import os
import pdb

import torch
import math
import transformers
import open_clip
import numpy as np

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, gather_object
from tqdm.auto import tqdm

import pdb as pdb_original
import sys

class ForkedPdb(pdb_original.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb_original.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
    
FILE_PATH = "/nfshomes/asarkar6/trinity/train_data/"
logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/nfshomes/asarkar6/trinity/",
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/nfshomes/asarkar6/trinity/",
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--make_plot",
        type=str,
        default="false",
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="/nfshomes/asarkar6/trinity/",
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    args = parser.parse_args()
    return args

def t2i_process_fn(batch):
    images = batch["image"]
    captions = batch["caption"]
    objects = batch["object"]

    for i in range(len(images)):
        try:
            images[i] = PIL.Image.open(os.path.join(FILE_PATH, images[i])).convert("RGB")
        except:
            print(f"corrupt at index {i}")
            images[i] = None
            captions[i] = ""
            objects[i] = ""
    
    batch["image"] = images
    batch["caption"] = captions
    batch["object"] = objects
    return batch

def return_trinity():
    # load datasets
    test_dataset = load_dataset('json', data_files = os.path.join(FILE_PATH, "metadata.jsonl"), split="train", num_proc=8)
    test_dataset = test_dataset.rename_column("file_name", "image")
    test_dataset = test_dataset.rename_column("prompt", "caption")

    # test_dataset.cast_column("image", Image(decode=False))
    test_dataset.set_transform(t2i_process_fn)

    return test_dataset
         
# computes 
def main():
    # get the user args
    args = parse_args()

    # set up data parallelism
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print("DTYPE", weight_dtype)

    # no need to train image encoders
    clip_model, preprocess = open_clip.create_model_from_pretrained('hf-hub:'+args.pretrained_model_name_or_path, precision="fp16", cache_dir=args.cache_dir)
    tokenizer = open_clip.get_tokenizer('hf-hub:'+args.pretrained_model_name_or_path)
    clip_model.requires_grad_(False)
    clip_model.to(accelerator.device, dtype=weight_dtype)

    def collate_fn(batch):
        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]
        objects = [item["object"] for item in batch]
        return {
            "images": images,
            "captions": captions,
            "objects": objects
        }

    # load the dataset
    test_dataloader = torch.utils.data.DataLoader(
            return_trinity(),
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )
    
    # Prepare everything with our `accelerator`.
    clip_model, test_dataloader = accelerator.prepare(clip_model, test_dataloader)
    clip_model.eval()

    # Infer!
    total_batch_size = args.train_batch_size * accelerator.num_processes
    args.num_train_epochs = math.ceil(len(return_trinity()) // total_batch_size)

    logger.info("***** Running inference *****")
    logger.info(f"  Num examples = {len(return_trinity())}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.num_train_epochs}")

    od = OrderedDict()
    al = OrderedDict()
    for epoch, batch in enumerate(tqdm(test_dataloader, desc="Inferring")):
        # get the images, captions and objects
        images = batch["images"]
        captions = batch["captions"]
        objects = batch["objects"]

        # get the sizes of objects
        l = []
        for idx, item1 in enumerate(objects):
            width, height = images[idx].size
            if item1 is not None:
                for _, item2 in enumerate(item1):
                    area = (item2["ymax"] - item2["ymin"]) * (item2["xmax"] - item2["xmin"])
                    if area > 0:
                        l.append(area/(width*height))
        od[epoch] = l

        # ForkedPdb().set_trace()

        #preprocessing
        # ForkedPdb().set_trace()
        images = torch.stack([preprocess(image).unsqueeze(0) for image in images], dim=0)
        captions = tokenizer(captions)

        # get the scores 
        with torch.no_grad(), torch.amp.autocast("cuda"):
            # get text features
            text_features = clip_model.encode_text(captions.to(accelerator.device))
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # get image features
            image_features = clip_model.encode_image(images.squeeze().to(accelerator.device))
            image_features /= image_features.norm(dim=-1, keepdim=True)

            al[epoch] = [(image_features[idx].reshape(1,-1)@text_features[idx].reshape(1,-1).T).detach().cpu().numpy() for idx in range(text_features.shape[0])]

    clip_scores_lst = gather_object(list(al.values()))
    flat_clip_scores = np.asarray(list(itertools.chain(*clip_scores_lst))).reshape(-1,1)

    obj_size_lst = gather_object(list(od.values()))
    flat_obj_sizes = np.asarray(list(itertools.chain(*obj_size_lst))).reshape(-1,1)

    # save the data
    np.save(os.path.join(args.output_dir, 'clip_scores.npy'), flat_clip_scores)
    np.save(os.path.join(args.output_dir, 'obj_sizes.npy'), flat_obj_sizes)

    ForkedPdb().set_trace()

    if args.make_plot == "true":
        # make histogram for clip scores
        plt.hist(flat_clip_scores, bins=50, alpha=0.7)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of CLIP scores")
        plt.savefig(os.path.join(args.output_dir, "clip_distribution.png"))

        # make histogram for object sizes
        plt.hist(obj_size_lst, bins=50, alpha=0.7)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Object sizes")
        plt.savefig(os.path.join(args.output_dir, "obj_size_distribution.png"))
        plt.show()

    accelerator.end_training()

        
if __name__ == "__main__":
    main()
