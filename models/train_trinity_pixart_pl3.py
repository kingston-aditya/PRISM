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

from pipeline1 import EncoderModel, ProjectLayer
from trinity_dataloader import PixartTrainDataset_pl3


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

# def check_count(batch, count):
#     N, _, tok_len, embed_size = batch.shape
#     # pad the images with 0s if object images is less than 3
#     k = 0
#     temp_prompt = []
#     for i in range(len(count)):
#         if count[i] != 0:
#             temp_tensor = torch.zeros(1, int(3 - count[i]), tok_len, embed_size)
#             fin_tensor_0 = torch.cat((temp_tensor, batch[k:k+count[i],:,:,:].transpose(0,1)), dim=1) 
#         else:
#             temp_tensor = torch.zeros(1, int(3 - count[i]), tok_len, embed_size)
#             fin_tensor_0 = temp_tensor 
#         temp_prompt.append(fin_tensor_0.reshape(1, 3*tok_len, embed_size))
#         k += count[i]
#     return torch.cat(temp_prompt, dim=0)

def read_Trinity_dataset():
    # read multiple files
    json_obj = {"image":[], "prompt":[], "object":[]}

    for name in glob.glob("/data/home/saividyaranya/PRISM/cached_folder_real/metadata_folder_again/*.jsonl"):
        with open(os.path.join("/data/home/saividyaranya/PRISM/cached_folder_real/metadata_folder_again/", name), "r") as f:
            for line in f:
                try:
                    temp = json.loads(line.strip())
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


# encode object prompt - Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_object(batch, lab_batch, lab_atn_batch, img_encoders, img_tokenizers, text_encoder):
    object_embeds_list = []

    with torch.no_grad():
        # create the image embeddings
        for img_tokenizer, img_encoder in zip(img_tokenizers, img_encoders):
            try:
                # get object image embeddings
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
            object_embeds_list.append(img_embeds)

        # create the label text embeddings
        lab_prompt_embeds = text_encoder(torch.cat(lab_batch).to(text_encoder.device), attention_mask=torch.cat(lab_atn_batch).to(text_encoder.device))[0]

    # fix the image embeddings  
    # two image encoders - 257x1024 + 257x1664 = 257x2688
    object_embeds = torch.concat(object_embeds_list, dim=-1)
    bt_size, tok_len, ebd_sz = object_embeds.size()
    object_embeds = object_embeds.view(1, bt_size*tok_len, ebd_sz)

    # fix the text embeddings
    bt_size, tok_len, ebd_sz = lab_prompt_embeds.size()
    lab_prompt_embeds = lab_prompt_embeds.view(1, bt_size*tok_len, ebd_sz)
    return object_embeds, lab_prompt_embeds

# Encode text prompt - Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(input_ids, attn_mask, text_encoder):
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
    json_obj = read_Trinity_dataset()

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

    lora_config = LoraConfig(
        r=args.rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "proj",
            "linear",
            "linear_1",
            "linear_2",
            # "scale_shift_table",      # not available due to the implementation in huggingface/peft, working on it.
        ],
        use_dora = args.use_dora,
        use_rslora = args.use_rslora
    )

    # Move transformer, vae and text_encoder to device and cast to weight_dtype
    transformer.to(accelerator.device)
    
    def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
        if not isinstance(model, list):
            model = [model]
        for m in model:
            for param in m.parameters():
                # only upcast trainable parameters into fp32
                if param.requires_grad:
                    param.data = param.to(dtype)

    transformer = get_peft_model(transformer, lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(transformer, dtype=torch.float32)

    transformer.print_trainable_parameters() # coment this.. not required

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                # save transformer weights
                transformer_ = accelerator.unwrap_model(transformer)
                lora_state_dict = get_peft_model_state_dict(transformer_, adapter_name="default")
                StableDiffusionPipeline.save_lora_weights(os.path.join(output_dir, "transformer_lora"), lora_state_dict)
                # save weights in peft format to be able to load them back
                transformer_.save_pretrained(output_dir)

                for _, model in enumerate(models):
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            # load the LoRA into the model
            transformer_ = accelerator.unwrap_model(transformer)
            transformer_.load_adapter(input_dir, "default", is_trainable=True)

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    lora_layers = filter(lambda p: p.requires_grad, transformer.parameters())

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        prompt_embeds = torch.stack([item["prompt_embeds"] for item in batch]).squeeze()
        attn_mask = torch.stack([item["attn_mask"] for item in batch]).squeeze()
        object_prompt_embeds = [item["object_prompt_embeds"] for item in batch]
        filenames = [item["filenames"] for item in batch]
        label_attn_mask = [item["label_attn_mask"] for item in batch]
        object_label_embeds = [item["object_label_embeds"] for item in batch]
        return {
            "pixel_values": pixel_values,
            "prompt_embeds": prompt_embeds,
            "object_prompt_embeds": object_prompt_embeds,
            "object_label_embeds": object_label_embeds,
            "attn_mask": attn_mask,
            "label_attn_mask": label_attn_mask,
            "filenames": filenames
        } 
    
    # DataLoaders creation.
    bgs = Image.open(os.path.join(args.bg_dir, np.random.choice(os.listdir(args.bg_dir))))
    precomputed_dataset = PixartTrainDataset_pl3(json_obj, args, bgs, max_length, text_tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # load the transformer
    trinity = EncoderModel(4096, 4096, num_blocks=args.blocks)
    align_trinity = EncoderModel(4096, 4096, num_blocks=args.blocks)
    proj_layer = ProjectLayer(4096, 2688)

    params_to_optimize = list(lora_layers) + list(trinity.parameters()) + list(align_trinity.parameters()) + list(proj_layer.parameters())
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # load trinity to cuda
    trinity.to(accelerator.device)
    proj_layer.to(accelerator.device)
    align_trinity.to(accelerator.device)

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler, trinity, align_trinity, proj_layer = accelerator.prepare(transformer, optimizer, train_dataloader, lr_scheduler, trinity, align_trinity, proj_layer)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

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
        all_pths = glob.glob(os.path.join(args.output_dir, "pixart-pl3-checkpoint-*"))
        if len(all_pths) != 0:
            path_name = sorted(all_pths, key=lambda x: int(x.split('-')[-1].split('.')[0]))[-1]
            accelerator.print(f"Resuming from checkpoint {path_name}")
            # load the transformer
            accelerator.load_state(os.path.join(args.output_dir, path_name))

            # load the trinity and proj layer
            trinity.load_state_dict(torch.load(os.path.join(os.path.join(args.output_dir, path_name), "trinity_checkpoint"+".pt"), weights_only=True))
            proj_layer.load_state_dict(torch.load(os.path.join(os.path.join(args.output_dir, path_name), "proj_checkpoint"+".pt"), weights_only=True))
            align_trinity.load_state_dict(torch.load(os.path.join(os.path.join(args.output_dir, path_name), "altrinity_checkpoint"+".pt"), weights_only=True))

            if args.mixed_precision == "fp16":
                models = [trinity, proj_layer, align_trinity]
                cast_training_params(models, dtype=torch.float32)

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
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Iterations",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    transformer.train()
    trinity.train()
    proj_layer.train()
    align_trinity.train()

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer), accelerator.accumulate(trinity), accelerator.accumulate(align_trinity), accelerator.accumulate(proj_layer):
                # encode prompts
                prompts = batch["prompt_embeds"]
                prompt_attention_mask = batch["attn_mask"]
                prompt_embeds = encode_prompt(prompts, prompt_attention_mask, text_encoder)

                # encode object images
                object_prompt_embeds = []
                object_label_embeds = []
                flag = 0
                for _, (item, litem, aitem) in enumerate(zip(batch["object_prompt_embeds"], batch["object_label_embeds"], batch["label_attn_mask"])):
                    if len(item) > 0:
                        try:
                            encoded_object, lab_prompt_embeds = encode_object(item, litem, aitem, img_encoders, img_tokenizers, text_encoder)

                            tok_sz = encoded_object.shape[-2]//len(item)
                            lab_tok_sz = lab_prompt_embeds.shape[-2]//len(item)
                            if len(item) == 2:
                                encoded_object = torch.cat((encoded_object, encoded_object[0,:tok_sz,:].unsqueeze(0)), dim=-2)
                                lab_prompt_embeds = torch.cat((lab_prompt_embeds, lab_prompt_embeds[0,:lab_tok_sz,:].unsqueeze(0)), dim=-2)
                            elif len(item) == 1:
                                encoded_object = torch.cat((encoded_object, encoded_object[0,:tok_sz,:].unsqueeze(0), encoded_object[0,:tok_sz,:].unsqueeze(0)), dim=-2)
                                lab_prompt_embeds = torch.cat((lab_prompt_embeds, lab_prompt_embeds[0,:lab_tok_sz,:].unsqueeze(0), lab_prompt_embeds[0,:lab_tok_sz,:].unsqueeze(0)), dim=-2)
                            else:
                                pass

                            encoded_object = encoded_object.to(accelerator.device)
                            lab_prompt_embeds = lab_prompt_embeds.to(accelerator.device)
                        except Exception as e:
                            print("!!! PADDING being done !!!")
                            flag = 1
                    else:
                        print("Epsilon padding happening !!! L-1011")
                        encoded_object = []
                        flag = 1 

                    object_prompt_embeds.append(encoded_object)
                    object_label_embeds.append(lab_prompt_embeds)
                
                if flag == 1:
                    print("FLAG is 1.. padding about to happen but stopped!!")
                    continue
                
                object_prompt_embeds = torch.stack(object_prompt_embeds).squeeze()
                object_label_embeds = torch.stack(object_label_embeds).squeeze()

                # normalize everything
                object_prompt_embeds = object_prompt_embeds/torch.norm(object_prompt_embeds, p=2, dim=-1, keepdim=True)
                prompt_embeds = prompt_embeds/torch.norm(prompt_embeds, p=2, dim=-1, keepdim=True)
                object_label_embeds = object_label_embeds/torch.norm(object_label_embeds, p=2, dim=-1, keepdim=True)
                
                # project it to text space
                object_prompt_embeds = proj_layer(object_prompt_embeds)

                # align object images and labels
                object_embeds = align_trinity(object_label_embeds, object_prompt_embeds, 0, 0, typ=args.mask_typ)
                object_embeds = object_embeds/torch.norm(object_embeds, p=2, dim=-1, keepdim=True)
                
                prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)

                # get the trinity embeds
                # with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                trinity_embeds = trinity(prompt_embeds, object_embeds, 0, 0, typ=args.mask_typ)
                trinity_embeds = trinity_embeds/torch.norm(trinity_embeds, p=2, dim=-1, keepdim=True)
                
                # Convert images to latent space
                img_pixel_vals = batch["pixel_values"].to(accelerator.device)
                latents = vae.encode(img_pixel_vals.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

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

                # Prepare micro-conditions.
                added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
                if getattr(transformer, 'module', transformer).config.sample_size == 128:
                    resolution = torch.tensor([args.resolution, args.resolution]).repeat(bsz, 1)
                    aspect_ratio = torch.tensor([float(args.resolution / args.resolution)]).repeat(bsz, 1)
                    resolution = resolution.to(dtype=weight_dtype, device=latents.device)
                    aspect_ratio = aspect_ratio.to(dtype=weight_dtype, device=latents.device)
                    added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

                if torch.isnan(trinity_embeds).any():
                    print("!!! NAN INPUTS!!!")
                    continue

                # Predict the noise residual and compute loss
                model_pred = transformer(noisy_latents,
                                         encoder_hidden_states=trinity_embeds,
                                         encoder_attention_mask=prompt_attention_mask,
                                         timestep=timesteps,
                                         added_cond_kwargs=added_cond_kwargs).sample.chunk(2, 1)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
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
                        save_path = os.path.join(args.output_dir, f"pixart-pl3-checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        unwrapped_transformer = accelerator.unwrap_model(transformer, keep_fp32_wrapper=True)
                        # unwrapped_transformer = unwrapped_transformer.float()
                        lora_sd = get_peft_model_state_dict(unwrapped_transformer)

                        target_dtype = unwrapped_transformer.dtype
                        for k, v in lora_sd.items():
                            lora_sd[k] = v.to(target_dtype)

                        # save transformer lora weights
                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=lora_sd,
                            safe_serialization=True,
                        )

                        # save the rem 2 models
                        # proj_layer_ = accelerator.unwrap_model(proj_layer)
                        # trinity_ = accelerator.unwrap_model(trinity)
                        # align_trinity_ = accelerator.unwrap_model(align_trinity)
                        torch.save(proj_layer.state_dict(), os.path.join(save_path, "proj_checkpoint"+".pt"))
                        torch.save(trinity.state_dict(), os.path.join(save_path, "trinity_checkpoint"+".pt"))
                        torch.save(align_trinity.state_dict(), os.path.join(save_path, "altrinity_checkpoint"+".pt"))

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer, keep_fp32_wrapper=False)
        transformer.save_pretrained(args.output_dir)
        lora_state_dict = get_peft_model_state_dict(transformer)
        StableDiffusionPipeline.save_lora_weights(os.path.join(args.output_dir, "transformer_lora"), lora_state_dict)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)