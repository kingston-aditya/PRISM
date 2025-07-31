#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import shutil
from contextlib import nullcontext
from pathlib import Path
import traceback

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/PRISM2/")
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.models.unets.unet_2d_condition import QwenVL_SD15_UNet2DModel
from PIL import Image

import glob, json
import pdb as pdb_original
from pipeline1 import EncoderModel
from trinity_dataloader import SD15_Qwen2_TrainDataset
from safetensors.torch import load_file

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__, log_level="INFO")

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

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )

    parser.add_argument(
        "--pretrained_lmm_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Path to pretrained VAE model with better numerical stability.",
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
        "--wanna_bg",
        type=int,
        default=0,
        help=(
            "Do you wanna bg in training?"
        ),
    )

    parser.add_argument(
        "--wanna_trans",
        type=int,
        default=1,
        help=(
            "0 for having a not having transformer and 1 for having one."
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
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )

    parser.add_argument(
        "--valid_path_name", 
        type=str, 
        default="/nfshomes/asarkar6/aditya/PRISM/validation/", 
        help="Validation path name."
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
        "--training_stage",
        type=int,
        default=1,
        help="The column of the dataset containing a caption or a list of objects.",
    )

    parser.add_argument(
        "--blocks",
        type=int,
        default=4,
        help="The column of the dataset containing a caption or a list of objects.",
    )

    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
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
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--backup",
        type=str,
        default="/nfshomes/asarkar6/aditya/PRISM/backup/",
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
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
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
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
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--num_images_pp",
        type=int,
        default=9,
        help="Number of images per prompt.",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--valid_checkpointing",
        type=int,
        default=15,
        help="Do a validation pass.",
    )

    parser.add_argument(
        "--do_valid",
        type=str,
        default="False",
        help="do you wanna do validation?",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
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
        "More details here: https://huggingface.co/papers/2303.09556.",
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
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--enable_npu_flash_attention", action="store_true", help="Whether or not to use npu flash attention."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--debug_loss",
        action="store_true",
        help="debug loss for each image, if filenames are available in the dataset",
    )
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[
            f.lower() for f in dir(transforms.InterpolationMode) if not f.startswith("__") and not f.endswith("__")
        ],
        help="The image interpolation method to use for resizing images.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args

# Data preprocessing transformations
def transform_image(imgs, args):
    # Get the specified interpolation method from the args
    interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)

    # Raise an error if the interpolation method is invalid
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation mode {args.image_interpolation_mode}.")
    
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=interpolation),  # Use dynamic interpolation method
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    pixel_values = []
    for img in imgs:
        pixel_values.append(train_transforms(img))
    return torch.stack(pixel_values)

def read_Trinity_dataset():
    # read multiple files
    json_obj = {"image":[], "prompt":[], "object":[]}

    for name in glob.glob("/nfshomes/asarkar6/trinity/train_data/*.jsonl"):
        with open(os.path.join("/nfshomes/asarkar6/trinity/train_data/", name), "r") as f:
            for line in f:
                try:
                    temp = json.loads(line.strip())

                    # ignore corrupt images
                    if temp["file_name"] in ['/data/home/saividyaranya/PRISM/cached_folder_real/images_again/1500000.png', '/data/home/saividyaranya/PRISM/cached_folder_real/images_again/1500001.png', '/data/home/saividyaranya/PRISM/cached_folder_real/images_again/1500002.png']: 
                        continue
                    
                    # saves the image
                    json_obj["image"].append(temp["file_name"])
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

def custom_clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)

    if torch.isnan(clip_coef).any() or torch.isinf(clip_coef).any():
        raise ValueError("Gradients going nan !!!") 
    
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

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

    # modify the custom clip grad function
    accelerator.clip_grad_norm_ = custom_clip_grad_norm_

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load the UNET
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Load the Qwen Model
    qwen25 = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.pretrained_lmm_name, ignore_mismatched_sizes=True, cache_dir=args.cache_dir)
    qwen25.to(accelerator.device, dtype=weight_dtype)

    # Load the trinity
    if args.wanna_trans == 1:
        trinity = EncoderModel(qwen25.config.hidden_size, qwen25.config.hidden_size, num_blocks=args.blocks)
        trinity.to(accelerator.device, dtype=weight_dtype)
        trinity.requires_grad_(False)

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    qwen25.requires_grad_(False)
    
    # Load the transformer
    if args.wanna_trans == 1:
        transformer = QwenVL_SD15_UNet2DModel(qwen25, unet, trinity)
        del trinity
    else:
        transformer = QwenVL_SD15_UNet2DModel(qwen25, unet, trinity=None)

    transformer.requires_grad_(False)
    del unet, qwen25
    transformer.to(accelerator.device, dtype=torch.float32)

    # Load scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir)

    # Load the Autoencoder
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir)
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)

    # Load the qwen 2.5 autoprocessor
    processor = AutoProcessor.from_pretrained(args.pretrained_lmm_name, max_pixels = 512*28*28)
        
    if args.mixed_precision == "fp16":
        if args.wanna_trans == 1:
            models = [transformer.unet, transformer.trinity, transformer.linear]
        else:
            models = [transformer.unet, transformer.linear]
        
        # only upcast trainable parameters (LoRA) into fp32
        for m in models:
            for param in m.parameters():
                if param.requires_grad:
                    param.data = param.to(dtype=torch.float32)
        

    if args.gradient_checkpointing:
        transformer.unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if args.wanna_trans == 1:
        params_to_optimize = list(transformer.trinity.parameters()) + list(transformer.linear.parameters())
    else:
        params_to_optimize = list(transformer.linear.parameters())
    
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Get the specified interpolation method from the args
    # interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)
    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        prompts = [item["prompts"] for item in batch]
        object_prompt_embeds = [item["object_prompt_embeds"] for item in batch]
        filenames = [item["filenames"] for item in batch]
        return {
            "pixel_values": pixel_values,
            "prompts": prompts,
            "object_prompt_embeds": object_prompt_embeds,
            "filenames": filenames
        }
    
    # load the dataset
    json_obj = read_Trinity_dataset()

    # DataLoaders creation.
    bgs = Image.open(os.path.join(args.bg_dir, np.random.choice(os.listdir(args.bg_dir))))

    # change dataloader
    precomputed_dataset = SD15_Qwen2_TrainDataset(json_obj, args, bgs)
    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Decide which layers to train 
    if args.wanna_trans == 1:
        transformer.trinity.requires_grad_(True)

    transformer.linear.requires_grad_(True)
    transformer.unet.requires_grad_(False)

    # Prepare everything with our `accelerator`.
    train_dataloader, lr_scheduler = accelerator.prepare(train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training stage " + str(args.training_stage) + "*****")
    logger.info(f"  Num examples = {len(precomputed_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # resume from checkpoint
    if args.resume_from_checkpoint == "latest":
        all_pths_2 = glob.glob(os.path.join(args.output_dir, "sd15-qwen25-checkpoint-st2-*"))
        all_pths = glob.glob(os.path.join(args.output_dir, "sd15-qwen25-checkpoint-st1-*"))

        if args.training_stage == 1:
            final_pth = all_pths 
        else:
            final_pth = all_pths if len(all_pths_2)!=0 else all_pths_2

        if len(final_pth) != 0:
            path_name = sorted(final_pth, key=lambda x: int(x.split('-')[-1].split('.')[0]))[-1]
            accelerator.print(f"Resuming from checkpoint {path_name}")
            input_dir = os.path.join(args.output_dir, path_name)
            
            # load the unet
            # load the transformer (lmm, unet, trinity, linear)
            linear_weights = torch.load(os.path.join(input_dir, "proj_checkpoint.pt"))

            if args.wanna_trans == 1:
                trinity_weights = torch.load(os.path.join(input_dir, "trinity_checkpoint.pt"))
                transformer.trinity.load_state_dict(trinity_weights)

            transformer.linear.load_state_dict(linear_weights)
            
            global_step = int(path_name.split("-")[-1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            initial_global_step = 0
            first_epoch = 0
            global_step = 0
    else:
        initial_global_step = 0
        first_epoch = 0
        global_step = 0

    # load a new optimizer for 2nd training stage
    if args.training_stage == 2:
        # get the Lora Modules
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        transformer_ = transformer

        # Add adapter and make sure the trainable params are in float32.
        transformer_.unet.add_adapter(unet_lora_config)
        lora_layers = filter(lambda p: p.requires_grad, transformer_.unet.parameters())
        list_lora_layers = list(lora_layers)

        params_to_optimize = list_lora_layers + params_to_optimize
        
        optimizer = optimizer_cls(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # wrap it with ddp
        transformer, optimizer = accelerator.prepare(transformer_, optimizer)

        initial_global_step = 0
        first_epoch = 0    
    else:
        # wrap it with ddp
        transformer = accelerator.prepare(transformer)    

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    transformer.train()

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # transform the pixel values
                batch["pixel_values"] = transform_image(batch["pixel_values"], args)

                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # get the inputs for each model
                if args.training_stage == 1:
                    # Ensure consistent R1 across all processes
                    if accelerator.is_main_process:
                        R1_tensor = torch.tensor(np.random.randint(1, 3), device=accelerator.device)
                    else:
                        R1_tensor = torch.tensor(1, device=accelerator.device)

                    # Synchronize across processes if distributed
                    if accelerator.distributed_type != "NO":
                        torch.distributed.broadcast(R1_tensor, src=0)
                    
                    R1 = R1_tensor.item()

                    # prompt only as input
                    if R1 == 1:
                        text = ["Describe an image in detail with prompt: " + prompt for prompt in batch["prompts"]]

                        # load the processor
                        inputs = processor(
                            text=text,
                            padding="longest",
                            return_tensors="pt",
                        )

                        if not hasattr(inputs, "to"):
                            print("skipping bcz processer returned dict from R1 = 1")
                            continue

                    elif R1 == 2:
                        messages = [
                            [{
                            "role": "user",
                            "content":[
                                {
                                    "type": "image",
                                    "image": img
                                } for img in img_list] + [{"type": "text", "text": "Explain an image in detail with above object."}]
                            }] for img_list in batch["object_prompt_embeds"]
                        ]

                        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
                        image_inputs, video_inputs = process_vision_info(messages)

                        inputs = processor(
                            text=texts,
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )

                        if not hasattr(inputs, "to"):
                            print("skipping bcz processer returned dict From R1 = 2")
                            continue

                elif args.training_stage == 2:
                    messages = [
                        [{
                        "role": "user",
                        "content":[
                            {
                                "type": "image",
                                "image": img
                            } for img in img_list] + [{"type": "text", "text": "Explain an image in detail with above objects following the prompt:" + batch["prompts"][idx]}]
                        }] for idx, img_list in enumerate(batch["object_prompt_embeds"])
                    ]

                    text = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]

                    image_inputs, video_inputs = process_vision_info(messages)

                    inputs = processor(
                        text=text,
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )

                else:
                    raise ValueError("Wrong training stage input !!!")
                
                # try-except distributed block
                exception_flag = torch.tensor(0.0, device=accelerator.device)
                try:
                    inputs = inputs.to(device=latents.device, dtype=weight_dtype)
                    inputs = {f"lmm_{k}": v for k, v in inputs.items()} 
                except Exception as e:
                    exception_flag = torch.tensor(1.0, device=accelerator.device)
                    accelerator.print(f"Process {accelerator.process_index} caught exception: {e}")
                exception_flag = accelerator.gather(exception_flag)

                if torch.sum(exception_flag) > 0:
                    accelerator.print("Skipping step due to error in one or more processes.")
                    continue


                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                inputs["unet_hidden_states"] = noisy_latents
                inputs["unet_time_step"] = timesteps

                # pass it through transformer
                model_pred = transformer(**inputs)

                # checking nan or inf block
                nan_or_inf = torch.isnan(model_pred).any() or torch.isinf(model_pred).any()
                nan_or_inf_tensor = torch.tensor(nan_or_inf, device=accelerator.device, dtype=torch.float32)
                nan_or_inf_tensor_gathered = accelerator.gather(nan_or_inf_tensor)
                if torch.sum(nan_or_inf_tensor_gathered) > 0:
                    accelerator.print(f"Model predictions going nan or inf at step {step}")
                    continue

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://huggingface.co/papers/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                ## skip batch is corrupt
                # checking nan or inf block
                nan_or_inf = torch.isnan(loss).any() or torch.isinf(loss).any()
                nan_or_inf_tensor = torch.tensor(nan_or_inf, device=accelerator.device, dtype=torch.float32)
                nan_or_inf_sum = accelerator.gather(nan_or_inf_tensor)
                if torch.sum(nan_or_inf_sum) > 0:
                    accelerator.print(f"Loss going nan or inf at step {step}")
                    continue

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (transformer.parameters())

                    # try-except distributed block
                    exception_flag = torch.tensor(0.0, device=accelerator.device)
                    try:
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    except Exception as e:
                        exception_flag = torch.tensor(1.0, device=accelerator.device)
                        accelerator.print(f"Process {accelerator.process_index} caught exception: {e}")
                    exception_flag = accelerator.gather(exception_flag)

                    if torch.sum(exception_flag) > 0:
                        accelerator.print("Skipping step due to error in one or more processes.")
                        continue

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"sd15-qwen25-checkpoint-st1-{global_step}")

                        if args.training_stage == 2:
                            save_path = os.path.join(args.output_dir, f"sd15-qwen25-checkpoint-st2-{global_step}")
                        
                        # create a new folder save_path
                        os.makedirs(save_path, exist_ok=True)
                        
                        transformer_ = accelerator.unwrap_model(transformer)
                        if args.training_stage == 2:
                            unet_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(transformer_.unet)
                            )

                            StableDiffusionPipeline.save_lora_weights(
                                save_directory=save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )

                        # save these layers for safety
                        torch.save(transformer_.linear.state_dict(), os.path.join(save_path, "proj_checkpoint"+".pt"))

                        if args.wanna_trans == 1:
                            torch.save(transformer_.trinity.state_dict(), os.path.join(save_path, "trinity_checkpoint"+".pt"))

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)