# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Union
from PIL import Image

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPVisionModelWithProjection

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from peft import LoraConfig, get_peft_model_state_dict, get_peft_model, PeftModel
from tqdm.auto import tqdm

import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/PRISM/")
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, PixArtAlphaPipeline, Transformer2DModel
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version

import pdb as pdb_original
import glob, json
from itertools import product

from pipeline1 import EncoderModel, ProjectLayer
from trinity_dataloader import PixartInferDataset


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "/nfshomes/asarkar6/trinity/train_data/": ("image", "prompt","object")
}

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

# TODO: This function should be removed once training scripts are rewritten in PEFT
def text_encoder_lora_state_dict(text_encoder):
    state_dict = {}

    def text_encoder_attn_modules(text_encoder):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        attn_modules = []

        if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))

        return attn_modules

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict

# def dynamic_collate(batch):
#     batch_x = []
#     for item in batch:
#         batch_x.append(item)
#     return batch_x

# def dynamic_collate_1(batch):
#     batch_x = {"image":[], "prompt":[]}
#     for item in batch:
#         batch_x["image"].append(item["image"])
#         batch_x["prompt"].append(item["prompt"])
#     return batch_x

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
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--bg_dir",
        type=str,
        default="/nfshomes/asarkar6/aditya/PRISM/backgrounds/",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )

    parser.add_argument(
        "--backup",
        type=str,
        default="/nfshomes/asarkar6/aditya/PRISM/backup/",
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--wanna_bg",
        type=int,
        default=0,
        help=(
            "Do you wanna bg in training?"
        ),
    )

    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )

    parser.add_argument(
        "--output_image_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )

    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )

    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )

    parser.add_argument(
        "--object_column",
        type=str,
        default="object",
        help="The column of the dataset containing a caption or a list of objects.",
    )

    parser.add_argument(
        "--mask_typ",
        type=str,
        default="normal",
        help="The column of the dataset containing a caption or a list of objects.",
    )

    parser.add_argument(
        "--blocks",
        type=int,
        default=4,
        help="The column of the dataset containing a caption or a list of objects.",
    )

    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pixart-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--valid_path_name",
        type=str,
        default="/nfshomes/asarkar6/aditya/PRISM/validation/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/nfshomes/asarkar6/trinity/",
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help="Whether or not to use Dora. For more information, see"
        " https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig.use_dora"
    )
    parser.add_argument(
        "--use_rslora",
        action="store_true",
        default=False,
        help="Whether or not to use RS Lora. For more information, see"
        " https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig.use_rslora"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
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
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    parser.add_argument("--local-rank", type=int, default=-1)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args

# reads a batch of image-text pairs  
def read_eval_dataset(args):
    # read multiple files
    json_obj = {"image":[], "prompt":[], "object":[]}

    for name in glob.glob(os.path.join(args.valid_path_name, "*.jsonl")):
        with open(os.path.join(args.valid_path_name, name), "r") as f:
            for line in f:
                try:
                    temp = json.loads(line.strip())
                    # saves the text prompt
                    json_obj["prompt"].append(temp["prompt"])
                    # saves the object
                    if temp["object"] is not None:
                        json_obj["object"].append(temp["object"])
                    else:
                        json_obj["object"].append([])
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON for line: {line.strip()} with error: {e}")
    return json_obj


# encode object prompt - Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_object(batch, img_encoders, img_tokenizers):
    object_embeds_list = []

    # create the image embeddings
    with torch.no_grad():
        idx = 0
        for img_tokenizer, img_encoder in zip(img_tokenizers, img_encoders):
            try:
                img_inputs = img_tokenizer(
                    images = batch,
                    return_tensors="pt",
                )
                img_inputs = img_inputs.to(img_encoder.device)
                img_embeds = img_encoder(**img_inputs, output_hidden_states=True, return_dict=False,)

                img_embeds = img_embeds[-1][-2]
                bs_embed, seq_len, _ = img_embeds.shape
                img_embeds = img_embeds.view(bs_embed, seq_len, -1)
            except Exception as e:
                raise RuntimeError("Unable to read this !!!")

            # We are only ALWAYS interested in the pooled output of the final text encoder
            idx+=1
            object_embeds_list.append(img_embeds)

    # two image encoders - 257x1024 + 257x1664 = 257x2688
    prompt_embeds = torch.concat(object_embeds_list, dim=-1)
    bt_size, tok_len, ebd_sz = prompt_embeds.size()
    prompt_embeds = prompt_embeds.view(1,bt_size*tok_len,ebd_sz)
    return prompt_embeds

# Encode text prompt - Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(input_ids, attn_mask, text_encoder):
    if len(input_ids.size()) != 2:
        input_ids = input_ids.unsqueeze(0)
        attn_mask = attn_mask.unsqueeze(0)
    with torch.no_grad():
        prompt_embeds = text_encoder(input_ids.to(text_encoder.device), attention_mask=attn_mask.to(text_encoder.device))[0]
    return prompt_embeds

def multimodal_encode_prompt(prompt_embeds, object_prompt_embeds):
    # concat them across dim 1
    cat_prompt_embeds = torch.cat((prompt_embeds, object_prompt_embeds), dim=-2)
    return cat_prompt_embeds

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token).repo_id

    # See Section 3.1. of the paper.
    max_length = 120

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load the CLIPImageProcessors
    img_tokenizer_one = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=args.cache_dir)
    img_tokenizer_two = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", cache_dir=args.cache_dir)

    # load image encoders
    img_encoder_one = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=args.cache_dir)
    img_encoder_two = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", cache_dir=args.cache_dir)

    # no need to train image encoders
    img_encoder_one.requires_grad_(False)
    img_encoder_two.requires_grad_(False)
    img_encoder_one.to(accelerator.device, dtype=weight_dtype)
    img_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # b) get the tokenizers and encoders ready
    img_tokenizers = [img_tokenizer_one, img_tokenizer_two]
    img_encoders = [img_encoder_one, img_encoder_two]

    # Get the embeddings and unload models
    # Step 1: get the object images - 
    # a) read the images and count - store all image embeds
    json_obj = read_eval_dataset(args)

    # Load noise scheduler.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=weight_dtype, cache_dir=args.cache_dir)
    
    # load text encoders
    text_tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, torch_dtype=weight_dtype,cache_dir=args.cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, torch_dtype=weight_dtype,cache_dir=args.cache_dir)
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device)

    # load vae
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant, torch_dtype=weight_dtype,cache_dir=args.cache_dir)
    vae.requires_grad_(False)
    vae.to(accelerator.device)

    # load pixart - transformer
    # freeze parameters of models to save more memory
    transformer = Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype, cache_dir=args.cache_dir)
    transformer.requires_grad_(False)    
    
    # Freeze the transformer parameters before adding adapters
    for param in transformer.parameters():
        param.requires_grad_(False)

    transformer.to(accelerator.device)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    def collate_fn(batch):
        prompt_embeds = torch.stack([item["prompt_embeds"] for item in batch]).squeeze()
        attn_mask = torch.stack([item["attn_mask"] for item in batch]).squeeze()
        object_prompt_embeds = [item["object_prompt_embeds"] for item in batch]
        return {
            "prompt_embeds": prompt_embeds,
            "object_prompt_embeds": object_prompt_embeds,
            "attn_mask": attn_mask
        } 
    
    # DataLoaders creation.
    precomputed_dataset = PixartInferDataset(json_obj, args, max_length, text_tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # load the transformer
    trinity = EncoderModel(4096, 4096, num_blocks=args.blocks)
    proj_layer = ProjectLayer(4096, 2688)

    # Prepare everything with our `accelerator`.
    trinity, proj_layer = accelerator.prepare(trinity, proj_layer)

    # load trinity to cuda
    trinity.to(accelerator.device)
    proj_layer.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(precomputed_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # resume from checkpoint
    if args.resume_from_checkpoint == "latest":
        all_pths = glob.glob(os.path.join(args.output_dir, "pixart-checkpoint-*"))
        if len(all_pths) != 0:
            path_name = sorted(all_pths, key=lambda x: int(x.split('-')[-1].split('.')[0]))[-1]

            accelerator.print(f"Resuming from checkpoint {path_name}")

            # load the PEFT weights
            transformer = PeftModel.from_pretrained(transformer, path_name)
            pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, transformer=transformer, text_encoder=text_encoder, vae=vae, torch_dtype=weight_dtype, cache_dir=args.cache_dir)
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            del transformer
            torch.cuda.empty_cache()

            # load trinity and proj layer
            trinity.load_state_dict(torch.load(os.path.join(os.path.join(args.output_dir, path_name), "trinity_checkpoint"+".pt"), weights_only=True))
            proj_layer.load_state_dict(torch.load(os.path.join(os.path.join(args.output_dir, path_name), "proj_checkpoint"+".pt"), weights_only=True))
        else:
            raise RuntimeError("Checkpoint not available.")
    else:
        pass

    # evaluation mode
    trinity.eval()
    proj_layer.eval()

    for step, batch in enumerate(tqdm(train_dataloader, desc="Inferring")):
        # encode prompts
        prompts = batch["prompt_embeds"]
        prompt_attention_mask = batch["attn_mask"]
        prompt_embeds = encode_prompt(prompts, prompt_attention_mask, text_encoder)

        # encode object images
        object_prompt_embeds = []
        # print(len(batch["object_prompt_embeds"][0]))
        flag = 0
        for _, item in enumerate(batch["object_prompt_embeds"]):
            if len(item) > 0:
                try:
                    encoded_object = encode_object(item, img_encoders, img_tokenizers)
                    tok_sz = encoded_object.shape[-2]//len(item)
                    if len(item) == 2:
                        encoded_object = torch.cat((encoded_object, encoded_object[0,:tok_sz,:].unsqueeze(0)), dim=-2)
                    elif len(item) == 1:
                        encoded_object = torch.cat((encoded_object, encoded_object[0,:tok_sz,:].unsqueeze(0), encoded_object[0,:tok_sz,:].unsqueeze(0)), dim=-2)
                    else:
                        pass
                    encoded_object = encoded_object.to(accelerator.device)
                except Exception as e:
                    print("padding being done.")
                    flag = 1
            else:
                print("Epsilon padding happening !!! L-1011")
                encoded_object = []
                flag = 1 

            object_prompt_embeds.append(encoded_object)
        
        if flag == 1:
            print("FLAG is 1.. padding about to happen but stopped!!")
            continue
        
        object_prompt_embeds = torch.stack(object_prompt_embeds).squeeze()

        # get multimodal prompts
        txt_tok_len = prompt_embeds.shape[-2]
        img_tok_len = object_prompt_embeds.shape[-2]

        if len(object_prompt_embeds.size()) == 2:
            object_prompt_embeds = object_prompt_embeds.unsqueeze(0)

        # normalize everything
        object_prompt_embeds = object_prompt_embeds/torch.norm(object_prompt_embeds, p=2, dim=-1, keepdim=True)
        prompt_embeds = prompt_embeds/torch.norm(prompt_embeds, p=2, dim=-1, keepdim=True)
        
        # with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        object_prompt_embeds = proj_layer(object_prompt_embeds)
        
        prompt_embeds = multimodal_encode_prompt(prompt_embeds, object_prompt_embeds)
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        
        # pad attention mask to match the concatenated prompt size
        pad_size = prompt_embeds.shape[1] - max_length
        prompt_attention_mask = F.pad(batch["attn_mask"], (0, pad_size), mode='constant', value=0).to(accelerator.device)

        # get the trinity embeds
        # with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        trinity_embeds = trinity(prompt_embeds, prompt_embeds, txt_tok_len, img_tok_len, typ=args.mask_typ)
        trinity_embeds = trinity_embeds/torch.norm(trinity_embeds, p=2, dim=-1, keepdim=True)

        if torch.isnan(trinity_embeds).any():
            print("!!! NAN INPUTS!!!")
            continue

        # Predict the noise residual and compute loss
        # load the original Pixart alpha model
        images = pipeline(prompt_embeds=trinity_embeds, num_inference_steps=50, prompt_attention_mask=prompt_attention_mask, num_images_per_prompt=args.num_validation_images).images

        for p_idx, i_idx in product(range(prompt_embeds.shape[0]), range(args.num_validation_images)):
            idx = p_idx * args.num_validation_images + i_idx
            pdx = step * prompt_embeds.shape[0] + p_idx
            images[idx].save(os.path.join(args.output_image_dir, f"prompt{pdx}_img{i_idx}.png"))

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)