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
import os
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
from huggingface_hub import create_repo
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModel
from itertools import product

import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/PRISM/")
import diffusers
from diffusers import DiffusionPipeline
from diffusers.utils import check_min_version
from PIL import Image

import glob, json
import pdb as pdb_original
from pipeline1 import EncoderModel, ProjectLayer
from trinity_dataloader import SD15InferDataset
import sys

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
        "--output_img_dir",
        type=str,
        default="/nfshomes/asarkar6/aditya/gen_images/",
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
def encode_object(batch, img_encoder, img_tokenizer):
    # create the image embeddings
    with torch.no_grad():
        idx = 0
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
        except:
            raise Exception("padding being done at encoding objects.")

        # We are only ALWAYS interested in the pooled output of the final text encoder
        idx+=1

    # two image encoders - 257x1024 + 257x1664 = 257x2688
    bt_size, tok_len, ebd_sz = img_embeds.size()
    img_embeds = img_embeds.view(1,bt_size*tok_len,ebd_sz)
    return img_embeds

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

    # Load the tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, cache_dir=args.cache_dir)

    # Load the Image tokenizer and Image encoder
    image_tokenizer = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=args.cache_dir)
    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=args.cache_dir)

    # Load the encoders
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # load the transformer
    trinity = EncoderModel(768, 768, num_blocks=args.blocks)
    proj_layer = ProjectLayer(768, 1024)

    # requires grad is true
    trinity.requires_grad_(True)
    proj_layer.requires_grad_(True)

    # Get the specified interpolation method from the args
    # interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)
    def collate_fn(batch):
        prompt_embeds = torch.stack([item["prompt_embeds"] for item in batch])
        object_prompt_embeds = [item["object_prompt_embeds"] for item in batch]
        return {
            "prompt_embeds": prompt_embeds,
            "object_prompt_embeds": object_prompt_embeds,
        }
    
    # load the dataset
    json_obj = read_eval_dataset(args)

    # DataLoaders creation.
    precomputed_dataset = SD15InferDataset(json_obj, args, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # load trinity to cuda
    trinity.to(accelerator.device)
    proj_layer.to(accelerator.device)

    # Prepare everything with our `accelerator`.
    trinity, proj_layer = accelerator.prepare(trinity, proj_layer)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(precomputed_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    global_step = 0

    # resume from checkpoint
    if args.resume_from_checkpoint == "latest":
        all_pths = glob.glob(os.path.join(args.output_dir, "sd15-checkpoint-*"))
        if len(all_pths) != 0:
            path_name = sorted(all_pths, key=lambda x: int(x.split('-')[-1].split('.')[0]))[-1]
            accelerator.print(f"Resuming from checkpoint {path_name}")
            input_dir = os.path.join(args.output_dir, path_name)
            
            # load the unet
            pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant, torch_dtype=weight_dtype, cache_dir=args.cache_dir, safety_checker = None, requires_safety_checker = False)
            pipeline.load_lora_weights(os.path.join(args.output_dir, path_name))
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # load the models and upcast it to float32
            trinity.load_state_dict(torch.load(os.path.join(input_dir, "trinity_checkpoint"+".pt"), weights_only=True))
            proj_layer.load_state_dict(torch.load(os.path.join(input_dir, "proj_checkpoint"+".pt"), weights_only=True))
        else:
            raise RuntimeError("Checkpoint not available.")
        
        # epochs
        global_step = 0

    trinity.eval()
    proj_layer.eval()

    for step, batch in enumerate(tqdm(train_dataloader, desc="Inferring")):
        # Get the text embeddings for conditioning
        prompt_embeds = text_encoder(batch["prompt_embeds"].to(text_encoder.device), return_dict=False)[0]

        # Get the object embeddings for conditioning
        object_prompt_embeds = []
        flag = 0
        for _, item in enumerate(batch["object_prompt_embeds"]):
            if len(item) > 0:
                try:
                    encoded_object = encode_object(item, image_encoder, image_tokenizer)
                    
                    tok_sz = encoded_object.shape[-2]//len(item)
                    if len(item) == 2:
                        encoded_object = torch.cat((encoded_object, encoded_object[0,:tok_sz,:].unsqueeze(0)), dim=-2)
                    elif len(item) == 1:
                        encoded_object = torch.cat((encoded_object, encoded_object[0,:tok_sz,:].unsqueeze(0), encoded_object[0,:tok_sz,:].unsqueeze(0)), dim=-2)
                    else:
                        pass
                    
                    encoded_object = encoded_object.to(accelerator.device)
                except Exception as e:
                    print("!!! PADDING being done !!!")
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

        # normalize everything
        object_prompt_embeds = object_prompt_embeds/torch.norm(object_prompt_embeds, p=2, dim=-1, keepdim=True)
        prompt_embeds = prompt_embeds/torch.norm(prompt_embeds, p=2, dim=-1, keepdim=True)

        # project it to text space
        object_prompt_embeds = proj_layer(object_prompt_embeds)

        # get the trinity embeds
        # with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        if len(object_prompt_embeds.size()) == 2:
            object_prompt_embeds = object_prompt_embeds.unsqueeze(0)
        trinity_embeds = trinity(prompt_embeds, object_prompt_embeds, 0, 0, typ=args.mask_typ)
        trinity_embeds = trinity_embeds/torch.norm(trinity_embeds, p=2, dim=-1, keepdim=True)

        if torch.isnan(trinity_embeds).any():
            print("!!! NAN INPUTS!!!")
            continue

        # Predict the noise residual and compute loss
        # load the original Pixart alpha model
        images = pipeline(prompt_embeds=trinity_embeds, text_embeds=prompt_embeds, num_inference_steps=50, num_images_per_prompt=args.num_validation_images).images

        for p_idx, i_idx in product(range(prompt_embeds.shape[0]), range(args.num_validation_images)):
            idx = p_idx * args.num_validation_images + i_idx
            pdx = step * prompt_embeds.shape[0] + p_idx
            images[idx].save(os.path.join(args.output_img_dir, f"prompt{pdx}_img{i_idx}.png"))

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)