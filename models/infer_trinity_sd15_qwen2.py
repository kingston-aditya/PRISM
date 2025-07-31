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
from diffusers.models.unets.unet_2d_condition import QwenVL_SD15_pipeline
from itertools import product
from PIL import Image

import glob, json
import pdb as pdb_original
from pipeline1 import EncoderModel
from trinity_dataloader import SD15_Qwen2_InferDataset

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
        "--image_or_prompt",
        type=int,
        default=1,
        help=(
            "1 for prompt and 2 for image only condition."
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

    parser.add_argument(
        "--wanna_trans",
        type=int,
        default=1,
        help="0 for no transformer and 1 for a transformer",
    )

    parser.add_argument(
        "--output_img_dir",
        type=str,
        default="/nfshomes/asarkar6/aditya/gen_images/",
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


    # Load the Qwen Model
    qwen25 = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.pretrained_lmm_name, ignore_mismatched_sizes=True, cache_dir=args.cache_dir)
    qwen25.to(accelerator.device, dtype=weight_dtype)

    # Load the trinity
    if args.wanna_trans == 1:
        trinity = EncoderModel(qwen25.config.hidden_size, qwen25.config.hidden_size, num_blocks=args.blocks)
        trinity.to(accelerator.device, dtype=weight_dtype)
        trinity.requires_grad_(False)

    # freeze parameters of models to save more memory
    qwen25.requires_grad_(False)

    # Load the transformer
    if args.wanna_trans == 1:
        transformer = QwenVL_SD15_pipeline(qwen25, trinity)
        del trinity
    else:
        transformer = QwenVL_SD15_pipeline(qwen25, trinity=None)
    
    transformer.requires_grad_(False)

    del qwen25
    transformer.to(accelerator.device, dtype=torch.float32)

    # Load the qwen 2.5 autoprocessor
    processor = AutoProcessor.from_pretrained(args.pretrained_lmm_name, max_pixels = 512*28*28)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Get the specified interpolation method from the args
    # interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)
    def collate_fn(batch):
        prompts = [item["prompts"] for item in batch]
        object_prompt_embeds = [item["object_prompt_embeds"] for item in batch]
        return {
            "prompts": prompts,
            "object_prompt_embeds": object_prompt_embeds,
        }
    
    # load the dataset
    json_obj = read_eval_dataset(args)

    # DataLoaders creation.
    bgs = Image.open(os.path.join(args.bg_dir, np.random.choice(os.listdir(args.bg_dir))))
    # change dataloader
    precomputed_dataset = SD15_Qwen2_InferDataset(json_obj, args, bgs)
    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    if args.wanna_trans == 1:
        transformer.trinity.requires_grad_(False)

    transformer.linear.requires_grad_(False)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running Inference stage " + str(args.training_stage) + "*****")
    logger.info(f"  Num examples = {len(precomputed_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")

    # resume from checkpoint
    if args.resume_from_checkpoint == "latest":
        if args.training_stage == 1:
            final_pth = glob.glob(os.path.join(args.output_dir, "sd15-qwen25-checkpoint-st1-*"))
        elif args.training_stage == 2:
            final_pth = glob.glob(os.path.join(args.output_dir, "sd15-qwen25-checkpoint-st2-*"))
        else:
            raise ValueError("Not able to find weights!!")

        if len(final_pth) != 0:
            path_name = sorted(final_pth, key=lambda x: int(x.split('-')[-1].split('.')[0]))[-1]
            accelerator.print(f"Resuming from checkpoint {path_name}")
            input_dir = os.path.join(args.output_dir, path_name)
            
            # load the transformer (lmm, unet, trinity, linear)
            linear_weights = torch.load(os.path.join(input_dir, "proj_checkpoint.pt"))
            transformer.linear.load_state_dict(linear_weights)

            if args.wanna_trans == 1:
                trinity_weights = torch.load(os.path.join(input_dir, "trinity_checkpoint.pt"))
                transformer.trinity.load_state_dict(trinity_weights)
        else:
            raise ValueError("Please give me a path name for loading weights!!")
        
        # use a stable diffusion pipeline
        sd_model = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False)
        if args.training_stage == 2:
            sd_model.load_lora_weights(os.path.join(args.output_dir, path_name))
        sd_model.set_progress_bar_config(disable=True)

    # Prepare everything with our `accelerator`.
    transformer= accelerator.prepare(transformer)
    transformer.eval()

    for step, batch in enumerate(tqdm(train_dataloader, desc="Inferring..")):
        # get the inputs for each model
        if args.training_stage == 1:
            # Ensure consistent R1 across all processes
            if accelerator.is_main_process:
                R1_tensor = torch.tensor(args.image_or_prompt, device=accelerator.device)
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

            elif R1 == 2:
                messages = [
                    [{
                    "role": "user",
                    "content":[
                        {
                            "type": "image",
                            "image": img
                        } for img in img_list] + [{"type": "text", "text": "Explain an image in detail with above objects."}]
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
        
        inputs = inputs.to(device=accelerator.device, dtype=weight_dtype)
        inputs = {f"lmm_{k}": v for k, v in inputs.items()}

        # pass it through transformer
        encoder_hidden_states = transformer(**inputs)

        # get the images
        sd_model.to(accelerator.device)

        with torch.no_grad():
            images = sd_model(prompt_embeds=encoder_hidden_states, num_inference_steps=50, num_images_per_prompt=args.num_validation_images).images

        # save the images
        for p_idx, i_idx in product(range(len(batch["prompts"])), range(args.num_validation_images)):
            idx = p_idx * args.num_validation_images + i_idx
            pdx = step * len(batch["prompts"]) + p_idx
            images[idx].save(os.path.join(args.output_img_dir, f"prompt{pdx}_img{i_idx}.png"))

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)