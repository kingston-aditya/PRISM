import argparse
import os

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import SiglipModel

from peft import LoraConfig, set_peft_model_state_dict
from torchvision.datasets import ImageFolder


import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm

import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/PRISM/dreamengine/src/diffusers/src/")
import diffusers
from diffusers.models.transformers.transformer_sd3 import QwenVLSD3Transformer2DModel, SD3Transformer2DModel, QwenVLSD3_DirectMap_Transformer2DModel
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/PRISM/")
from dataloaders.coco_dataloader import MS_COCO, StreamingMS_COCO

from transformers import Qwen2VLForConditionalGeneration, SiglipProcessor, AutoProcessor
from connector import MultimodalFusionModel, MLPProjection


import wandb
from datasets import load_dataset


logger = get_logger(__name__)

import os
import json
import torch
from torchvision import transforms
import torch.nn.functional as F


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args_list,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
    )
    # pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    autocast_ctx = nullcontext()

    with autocast_ctx:
        if "prompt" in pipeline_args_list[0]:
            images = [pipeline.cfg_predict(**pipeline_args, generator=generator, guidance_scale=3.5).images[0] for pipeline_args in pipeline_args_list]
        else:
            images = [pipeline(**pipeline_args, generator=generator).images[0] for pipeline_args in pipeline_args_list]
    
    
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"

        if "prompt" in pipeline_args_list[0]:
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {pipeline_args_list[i]['prompt']}") for i, image in enumerate(images)
                    ]
                }
            )
        else:
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption="Reconstruction") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()

    return images

def load_sharded_model(config_path, index_path, bin_files_folder, qwenvl2_model, sd3_model, device='cpu'):
    """
    Loads a sharded Hugging Face model from multiple binary files.

    Args:
        config_path (str): Path to the model configuration JSON file.
        index_path (str): Path to the model index JSON file.
        bin_files_folder (str): Directory containing the binary model files.
        device (str): Device to load the model onto ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded model with weights.
    """
    # Step 1: Load the Model Configuration
    print("Loading model configuration...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize the model using the configuration
    print("Initializing the model based on the configuration...")
    model = QwenVLSD3_DirectMap_Transformer2DModel(qwenvl2_model, sd3_model)
    
    # Step 2: Load the Model Index
    print("Loading model index file...")
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    weight_map = index.get('weight_map', {})
    if not weight_map:
        raise ValueError("The index file does not contain a 'weight_map' key.")
    
    # Step 3: Organize Weights by Binary File
    print("Organizing weights by their respective binary files...")
    bins = {}
    for weight_name, bin_file in weight_map.items():
        bins.setdefault(bin_file, []).append(weight_name)
    
    # Initialize an empty state dictionary
    state_dict = {}
    
    # Step 4: Load Each Binary File and Extract Relevant Weights
    for bin_file, weight_names in bins.items():
        bin_path = os.path.join(bin_files_folder, bin_file)
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(f"Binary file not found: {bin_path}")
        
        print(f"Loading binary file: {bin_path}")
        bin_state = torch.load(bin_path, map_location=device)
        
        # Determine how the weights are stored in the binary file
        # Common scenarios:
        # a) The entire state_dict is stored directly
        # b) The state_dict is nested under a key like 'state_dict'

        if isinstance(bin_state, dict):
            if 'state_dict' in bin_state:
                partial_state = bin_state['state_dict']
            else:
                partial_state = bin_state

            # Extract only the weights relevant to this bin file
            for weight_name in weight_names:
                if weight_name in partial_state:
                    state_dict[weight_name] = partial_state[weight_name]
                else:
                    print(f"Warning: '{weight_name}' not found in '{bin_file}'.")
        else:
            raise ValueError(f"Unexpected format in binary file: {bin_file}")

    # Step 5: Load the Merged State Dictionary into the Model
    print("Loading weights into the model...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print("Warning: The following keys are missing in the state dictionary:")
        for key in missing_keys:
            print(f"  - {key}")
    if unexpected_keys:
        print("Warning: The following keys are unexpected in the state dictionary:")
        for key in unexpected_keys:
            print(f"  - {key}")
    
    # Transfer the model to the specified device
    print(f"Transferring the model to {device.upper()}...")
    model.to(device)

    print("Model loaded successfully.")
    return model

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    # model checkpoints
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the seedraw checkpoint to load.")
    parser.add_argument("--output_dir", type=str, default="sd3-dreambooth", help="Directory for model predictions and checkpoints.")

    parser.add_argument("--pretrained_diffusion_ckpt", type=str, required=True, help="Path to the pretrained diffusion checkpoint.")
    parser.add_argument("--pretrained_lmm_ckpt", type=str, required=True, help="Path to the pretrained LMM checkpoint.")

    parser.add_argument("--keep_context_embedder", action="store_true", help="Keep the original context embedder of MMDiT")

    parser.add_argument("--cfg_ratio", type=float, default=None)
    parser.add_argument("--random_vit_skip", action="store_true")

    parser.add_argument("--unfreeze_connector", action="store_true", help="Whether to unfreeze the connector.")


    # dataset 
    parser.add_argument("--dataset_config", type=str, default=None, help="The path to the dataset configuration file.")
    parser.add_argument("--cache_dir", type=str, default="./QwenVL-GEN/diffusers/examples/dreambooth/cache", help="The directory where the downloaded models and datasets will be stored.")

    # training
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible training.")
    parser.add_argument("--output_resolution", type=int, default=512, help="The resolution for input images.")
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop the input images.")
    parser.add_argument("--random_flip", action="store_true", help="Whether to randomly flip images horizontally.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size per device.")
    parser.add_argument("--sample_batch_size", type=int, default=4, help="Sampling batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates.")

    parser.add_argument("--lora_rank", type=int, default=16, help="Rank for Lora.")
    parser.add_argument("--use_lmm_attention_lora", action="store_true", help="Whether to use lora for LMM attention.")
    parser.add_argument("--use_dit_attention_lora", action="store_true", help="Whether to use lora for DiT attention.")
    parser.add_argument("--use_vit_attention_lora", action="store_true", help="Whether to use lora for ViT attention.")

    parser.add_argument("--print_params", action="store_true", help="Whether to print trainable params or not.")
    parser.add_argument("--use_lmm_mlp_lora", action="store_true", help="Use LMM lora on MLP.")
    parser.add_argument("--use_dit_mlp_lora", action="store_true", help="Use DIT lora on MLP.")
    parser.add_argument("--dit_lora_layers", type=str, default=None, help="Specify layers for DIT lora.")
    
    parser.add_argument("--unfreeze_lmm", action="store_true", help="Whether to unfreeze the language model head.")
    parser.add_argument("--unfreeze_lmm_layers", type=str, default=None, help="The specific layers to unfreeze in the lmm.")
    parser.add_argument("--unfreeze_dit", action="store_true", help="Whether to unfreeze the DIT.")
    parser.add_argument("--unfreeze_dit_layers", type=str, default=None, help="The specific layers to unfreeze in the DIT.")
    parser.add_argument("--unfreeze_adapter", action="store_true", help="Whether to unfreeze the adapter.")
    parser.add_argument("--unfreeze_dit_condition_branch", action="store_true", help="Whether to unfreeze the DIT condition branch.")
    parser.add_argument("--unfreeze_dit_embed", action="store_true", help="Whether to unfreeze the DIT time embed.")
    parser.add_argument("--unfreeze_dit_output_layer", action="store_true", help="Whether to unfreeze DIT output layer.")
    parser.add_argument("--linear_alignment", action="store_true", help="Whether to use linear alignment.")
    parser.add_argument("--lmm_output_layer_index", type=int, default=-1, help="Layer index for LMM output.")
    parser.add_argument("--input_resolution", type=int, default=512, help="Resolution for input images in training/validation dataset.")
    parser.add_argument("--do_lmm_post_norm", action="store_true", help="Whether to use LMM post normalization.")
    parser.add_argument("--max_sequence_length", type=int, default=None, help="Maximum sequence length for T5 text encoder.")

    parser.add_argument("--num_latents", type=int, default=256, help="Maximum sequence length for T5 text encoder.")

    parser.add_argument("--structure", type=str, default=None)

    # evaluation
    parser.add_argument("--validation_prompts", type=str, default=None, help="Prompts used in validation for learning verification.")
    parser.add_argument("--validation_images", type=str, default="./diffusers/examples/dreambooth/syncdog_val.png", help="Image used during validation.")
    parser.add_argument("--num_validation_images", type=int, default=1, help="Number of images generated during validation with `validation_prompt`.")
    parser.add_argument("--validation_steps", type=int, default=500, help="Run DreamBooth validation every X steps.")

    parser.add_argument("--validation_edit_prompts", type=str, default=None)
    parser.add_argument("--validation_edit_images", type=str, default=None)

    

    # others
    parser.add_argument("--checkpoints_total_limit", type=int, default=10, help="Max number of checkpoints to store.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps to accumulate before backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")

    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Scheduler type for learning rate.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of cycles for cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=16, help="Number of subprocesses for data loading.")
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"], help="Scheme for weighting.")
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Mean for logit_normal weighting scheme.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="Standard deviation for logit_normal weighting scheme.")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale for mode weighting scheme.")

    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 for Adam/Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 for Adam/Prodigy optimizers.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4, help="Weight decay for UNet parameters.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam/Prodigy optimizers.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard log directory.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Integration to report logs to; can be 'tensorboard', 'wandb', or 'comet_ml'.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mode for mixed precision training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank.")

    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class QwenVLSD3_Perceiver_Model(QwenVLSD3_DirectMap_Transformer2DModel):
    def __init__(self, qwenvl_model, sd3_dit_model, num_latents=256, mlp_dim=4096, linear_alignment=False, lmm_output_layer_index=-1, do_lmm_post_norm=False):
        # initializes the base parameters of parent class
        super().__init__(qwenvl_model, sd3_dit_model, mlp_dim, linear_alignment, lmm_output_layer_index, do_lmm_post_norm)

        self.siglip_model = SiglipModel.from_pretrained("google/siglip-base-patch16-224", torch_dtype=torch.bfloat16)

        self.connector = MultimodalFusionModel(self.siglip_model.config.vision_config.hidden_size, sd3_dit_model.config.caption_projection_dim, num_latents=num_latents)
        self.small_mlp =  MLPProjection(sd3_dit_model.config.caption_projection_dim, sd3_dit_model.config.caption_projection_dim)

        self.num_latents = num_latents

        # freeze stuff
        for param in self.dit.parameters():
            param.requires_grad = False
        for param in self.lmm.parameters():
            param.requires_grad = False
        for param in self.input_embeds_align_mlp.parameters():
            param.requires_grad = False
        for param in self.condition_embeds_align_mlp.parameters():
            param.requires_grad = False
        for param in self.siglip_model.parameters():
            param.requires_grad = False

        # unfreeze stuff
        for param in self.connector.parameters():
            param.requires_grad = True
        
        for param in self.small_mlp.parameters():
            param.requires_grad = True
            
    
    def forward(self, lmm_input_ids, lmm_attention_mask, lmm_pixel_values, lmm_image_grid_thw, siglip_text_inputs, siglip_image_inputs, dit_hidden_states, dit_time_step, dit_text_condition=None, pooled_dit_text_condition=None, vit_skip_ratio=None):
        # 1. Get frozen LMM states
        with torch.no_grad():
            # pad the tokens
            batch_size, current_input_len = lmm_input_ids.shape
            target_seq_len = self.num_latents  # e.g., 256

            # Fetch the model's official padding token ID
            pad_token_id = getattr(self.lmm.config, "eos_token_id", 151645)

            # 2. Compare inputs vs PR target capacity and align
            if current_input_len >= target_seq_len:
                # --- CONDITION 2: TRUNCATE ---
                # Slice the input tokens and mask from the right side down to target capacity
                lmm_input_ids = lmm_input_ids[:, :target_seq_len]
                lmm_attention_mask = lmm_attention_mask[:, :target_seq_len]
            else:
                # --- CONDITION 3: PAD ---
                padding_needed = target_seq_len - current_input_len
                
                # Create the text input padding tensor filled with the pad token ID
                input_ids_padding = torch.full(
                    (batch_size, padding_needed), 
                    pad_token_id, 
                    dtype=lmm_input_ids.dtype, 
                    device=lmm_input_ids.device
                )
                
                # Create the attention mask padding tensor filled with 0s (ignore slots)
                attention_mask_padding = torch.zeros(
                    (batch_size, padding_needed), 
                    dtype=lmm_attention_mask.dtype, 
                    device=lmm_attention_mask.device
                )
                
                # Concatenate along the 1D sequence dimension
                lmm_input_ids = torch.cat([lmm_input_ids, input_ids_padding], dim=1)
                lmm_attention_mask = torch.cat([lmm_attention_mask, attention_mask_padding], dim=1)

            lmm_outputs_last_hidden_state = self.lmm(
                input_ids=lmm_input_ids,
                attention_mask=lmm_attention_mask,
                pixel_values=lmm_pixel_values,
                image_grid_thw=lmm_image_grid_thw,
                output_hidden_states=True
            )['hidden_states'][self.lmm_output_layer_index]

            dit_encoder_hidden_states_proj = self.input_embeds_align_mlp(lmm_outputs_last_hidden_state)
            dit_encoder_hidden_states_pooled_proj = self.condition_embeds_align_mlp(lmm_outputs_last_hidden_state.mean(dim=1))

        # 2. Get frozen SigLIP states
        with torch.no_grad():
            text_embeddings = self.siglip_model.text_model(**siglip_text_inputs).last_hidden_state
            embedding_dim = text_embeddings.shape[-1]
            batch_size = text_embeddings.shape[0]
            
            num_objects_per_sample = int(siglip_image_inputs["pixel_values"].shape[0]//batch_size)
            vision_outputs = self.siglip_model.vision_model(pixel_values=siglip_image_inputs["pixel_values"])
            flat_patch_embeddings = vision_outputs.last_hidden_state 
            patch_seq_len = flat_patch_embeddings.shape[1]
            
            batch_visual_embeddings = flat_patch_embeddings.view(
                batch_size, num_objects_per_sample, patch_seq_len, embedding_dim
            )
            batch_visual_embeddings = batch_visual_embeddings.view(
                batch_size, num_objects_per_sample * patch_seq_len, embedding_dim
            )
            multimodal_embeddings = torch.cat([batch_visual_embeddings, text_embeddings], dim=1)

        # 3. TRAINABLE TRACK (Keep outside of no_grad!)
        # Gradients will flow through these two models perfectly
        dit_encoder_conn_proj = self.connector(multimodal_embeddings)
        dit_encoder_conn_pooled_proj = self.small_mlp(dit_encoder_conn_proj.mean(dim=1))
        
        # 4. Get frozen DiT time embedding structures
        with torch.no_grad():
            dit_time_embed = self.dit.time_text_embed.time_proj(dit_time_step)
            dit_timesteps_emb = self.dit.time_text_embed.timestep_embedder(dit_time_embed.to(dtype=dit_encoder_hidden_states_pooled_proj.dtype))
            
        # 5. Build dynamic combinations 
        # (Mixing gradient tracking vectors with detached time vectors is perfectly fine!)
        dit_concat_condition = dit_encoder_hidden_states_pooled_proj + dit_timesteps_emb
        dit_concat_condition_connector = dit_encoder_conn_pooled_proj + dit_timesteps_emb

        # 6. Pass through frozen DiT forward passes
        # PyTorch will still trace gradients backward through your encoder inputs!
        with torch.no_grad():
            dit_model_pred = self.dit.forward_with_lmm_encoder(
                            hidden_states=dit_hidden_states,
                            lmm_encoder_hidden_states=dit_encoder_hidden_states_proj,
                            lmm_pooled_projections=dit_concat_condition,
                            dit_text_condition=dit_text_condition,
                        )

        # CRITICAL: This specific forward call must track gradients through dit_encoder_conn_proj 
        # and dit_concat_condition_connector, so do NOT put it inside torch.no_grad()
        dit_model_pred_conn = self.dit.forward_with_lmm_encoder(
                        hidden_states=dit_hidden_states,
                        lmm_encoder_hidden_states=dit_encoder_conn_proj,
                        lmm_pooled_projections=dit_concat_condition_connector,
                        dit_text_condition=dit_text_condition,
                    )
        
        # 7. Compute Loss Functions outside of no_grad
        loss1 = F.mse_loss(dit_encoder_conn_proj, dit_encoder_hidden_states_proj)
        loss2 = F.mse_loss(dit_model_pred[0], dit_model_pred_conn[0])
        loss3 = F.mse_loss(dit_encoder_conn_pooled_proj, dit_encoder_hidden_states_pooled_proj)
        
        return 0.4 * loss1 + 0.4 * loss3 + 0.2 * loss2
    
import pdb as pdb_original

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

def print_trainable_parameters(model):
    print("=" * 40)
    print("TRAINABLE LAYERS / PARAMETERS:")
    print("=" * 40)
    
    trainable_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        total_count += param.numel()
        if param.requires_grad:
            print(f":) TRAINABLE: {name} | Shape: {list(param.shape)} | Dtype: {param.dtype}")
            trainable_count += param.numel()
            
    print("=" * 40)
    print(f"Trainable Params: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.2f}%)")
    print("=" * 40)

def main(args):

    # prepare accelerate distributed training

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
        dataloader_config=dataloader_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    
        
    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_diffusion_ckpt, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_diffusion_ckpt,
        subfolder="vae",
    )

    processor = AutoProcessor.from_pretrained(args.pretrained_lmm_ckpt, max_pixels = args.input_resolution*28*28)
    qwenvl2_model = Qwen2VLForConditionalGeneration.from_pretrained(args.pretrained_lmm_ckpt, torch_dtype=torch.bfloat16)
    qwenvl2_model = qwenvl2_model.to("cuda")
    print("loaded Qwen")

    sd3_model = SD3Transformer2DModel.from_pretrained(args.pretrained_diffusion_ckpt, subfolder="transformer", torch_dtype=torch.bfloat16)
    sd3_model = sd3_model.to("cuda")
    print("loaded SD3")

    # load the siglip processor
    siglip_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

    if args.structure=="direct":
        print("use direct structure")
        # transformer = QwenVLSD3_DirectMap_Transformer2DModel(qwenvl2_model, sd3_model, linear_alignment=args.linear_alignment, lmm_output_layer_index=args.lmm_output_layer_index, do_lmm_post_norm=args.do_lmm_post_norm)
        transformer = QwenVLSD3_Perceiver_Model(qwenvl2_model, sd3_model, num_latents=args.num_latents, linear_alignment=args.linear_alignment, lmm_output_layer_index=args.lmm_output_layer_index, do_lmm_post_norm=args.do_lmm_post_norm)
    else:
        transformer = QwenVLSD3Transformer2DModel(qwenvl2_model, sd3_model, linear_alignment=args.linear_alignment, lmm_output_layer_index=args.lmm_output_layer_index, do_lmm_post_norm=args.do_lmm_post_norm)

    del qwenvl2_model, sd3_model
    transformer.to(accelerator.device)
    transformer.requires_grad_(False)

    # setup lora
    dit_lora_target_modules = []
    lmm_lora_target_modules = []


    if args.use_vit_attention_lora:
        lmm_lora_target_modules += [
            "attn.qkv",
        ]

    if args.use_dit_attention_lora:
        dit_lora_target_modules += [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]

    if args.use_dit_mlp_lora:
        dit_lora_target_modules += [
                "ff.net.0.proj",
                "ff.net.2.proj",
                "ff_context.net.0.proj",
                "ff_context.net.2.proj"
    ]





    if len(dit_lora_target_modules) > 0:
        logger.info(f"Adding DiT Lora with rank {args.lora_rank} to the following modules: {dit_lora_target_modules}")
        dit_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2*args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=dit_lora_target_modules,
        )
        transformer.dit.add_adapter(dit_lora_config)
   
    if args.use_lmm_attention_lora:
        lmm_lora_target_modules += [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]

    if args.use_lmm_mlp_lora:
        lmm_lora_target_modules += [
                "mlp.down_proj",
                "mlp.gate_proj",
                "mlp.up_proj"
                ]

    if len(lmm_lora_target_modules) > 0:
        logger.info(f"Adding LMM Lora with rank {args.lora_rank} to the following modules: {lmm_lora_target_modules}")
        lmm_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2*args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=lmm_lora_target_modules,
        )

        transformer.lmm.add_adapter(lmm_lora_config)
    
    # resume from checkpoint

    if args.resume_from_checkpoint:
        logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        transformer = load_sharded_model(
            os.path.join(args.resume_from_checkpoint,"config.json"),
            os.path.join(args.resume_from_checkpoint,"diffusion_pytorch_model.bin.index.json"),
            os.path.join(args.resume_from_checkpoint,),
            qwenvl2_model,
            sd3_model,
            "cpu",
        )
    
    params_to_optimize = []

    # unfreeze parameters

    if args.unfreeze_dit:
        transformer.dit.requires_grad_(True)
    if args.unfreeze_dit_layers:
        layers_to_unfreeze = [int(i) for i in args.unfreeze_dit_layers.split(",")]
        for layer_index in layers_to_unfreeze:
            transformer.dit.transformer_blocks[layer_index].requires_grad_(True)

    
    if args.unfreeze_lmm:
        transformer.lmm.requires_grad_(True)
    if args.unfreeze_lmm_layers:
        layers_to_unfreeze = [int(i) for i in args.unfreeze_lmm_layers.split(",")]
        for layer_index in layers_to_unfreeze:
            transformer.lmm.model.layers[layer_index].requires_grad_(True)

    if args.unfreeze_dit_output_layer:
        transformer.dit.norm_out.requires_grad_(True)
        transformer.dit.proj_out.requires_grad_(True)
    
    if args.unfreeze_dit_embed:
        transformer.dit.time_text_embed.requires_grad_(True)
        transformer.dit.context_embedder.requires_grad_(True)
        transformer.dit.pos_embed.requires_grad_(True)
        

    if args.unfreeze_adapter:
        transformer.input_embeds_align_mlp.requires_grad_(True)
        transformer.condition_embeds_align_mlp.requires_grad_(True)
        transformer.norm_lmm_out.requires_grad_(True)
        transformer.pooled_proj.requires_grad_(True)

    ## Connector
    if args.unfreeze_connector:
        transformer.connector.requires_grad_(True)
        transformer.small_mlp.requires_grad_(True)
        transformer.siglip_model.requires_grad_(False)

    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    params_to_optimize.append({"params": transformer_trainable_parameters, "lr": args.learning_rate})

    
    # Calculate the number of parameters to train and all parameters

    num_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in transformer.parameters())

    
    logger.info(f"Number of parameters: {num_params}")
    logger.info(f"Number of parameters to train: {num_trainable_params}")

    # ratio
    logger.info(f"Trainable parameters ratio: {num_trainable_params / num_params:.5f}")

    # setting weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=torch.float32)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        # 1. Safely unwrap the model to avoid DDP / DeepSpeed wrapper namespaces
        # (Assuming you pass your main QwenVLSD3_Perceiver_Model to accelerate.prepare)
        for model in models:
            # Unwrap model if it is a wrapper
            unwrapped_model = accelerator.unwrap_model(model)
            
            # Double check that we are targeting your specific custom model class
            if hasattr(unwrapped_model, "connector") and hasattr(unwrapped_model, "dit"):
                print("Saving trainable connector, small_mlp, and dit parameters...")
                
                # --- Save Connector ---
                connector_state = unwrapped_model.connector.state_dict()
                torch.save(connector_state, os.path.join(output_dir, "connector.bin"))
                
                # --- Save Small MLP ---
                small_mlp_state = unwrapped_model.small_mlp.state_dict()
                torch.save(small_mlp_state, os.path.join(output_dir, "small_mlp.bin"))
                
                # --- Save DiT (Only the trainable parts, or all of it if needed) ---
                # trainable_keys = {
                #     name for name, param in unwrapped_model.dit.named_parameters() 
                #     if param.requires_grad
                # }
                # dit_state = {
                #     k: v for k, v in unwrapped_model.dit.state_dict().items() 
                #     if k in trainable_keys
                # }
                
                # # If nothing in DiT is trainable, save the whole state dict for structural safety,
                # # or skip it to save space.
                # if len(dit_state) == 0:
                #     print("DiT is completely frozen. Saving full DiT structure...")
                #     dit_state = unwrapped_model.dit.state_dict()
                    
                # torch.save(dit_state, os.path.join(output_dir, "dit.bin"))
                
        # 2. POP the weights array so accelerate knows you've handled model saving manually
        # and doesn't write the massive whole-model (LMM + Siglip) checkpoints
        while len(weights) > 0:
            weights.pop()

    def load_model_hook(models, input_dir):
        for model in models:
            unwrapped_model = accelerator.unwrap_model(model)
            
            if hasattr(unwrapped_model, "connector") and hasattr(unwrapped_model, "dit"):
                print("Restoring connector, small_mlp, and dit weights from checkpoint...")
                
                # Load Connector
                connector_path = os.path.join(input_dir, "connector.bin")
                if os.path.exists(connector_path):
                    unwrapped_model.connector.load_state_dict(torch.load(connector_path, map_location="cpu"))
                    
                # Load Small MLP
                small_mlp_path = os.path.join(input_dir, "small_mlp.bin")
                if os.path.exists(small_mlp_path):
                    unwrapped_model.small_mlp.load_state_dict(torch.load(small_mlp_path, map_location="cpu"))
                    
                # Load DiT
                dit_path = os.path.join(input_dir, "dit.bin")
                if os.path.exists(dit_path):
                    # Using strict=False in case you only saved unfrozen/LoRA components of the DiT
                    unwrapped_model.dit.load_state_dict(torch.load(dit_path, map_location="cpu"), strict=False)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


       
    # set up optimizer
    optimizer = torch.optim.AdamW(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    
    # set up dataset
    DATA_DIR = "/fs/cml-datasets/coco/"
    # train_dataset = MS_COCO(dataset_path = os.path.join(DATA_DIR, "annotations", "captions_val2017.json"))
    
    train_dataset = MS_COCO(dataset_path=os.path.join(DATA_DIR, "annotations", "captions_val2017.json"))
    
    # data loader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn,
        num_workers=8,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    estimated_dataset_size = int(10e5)

    num_update_steps_per_epoch = math.ceil(estimated_dataset_size / args.gradient_accumulation_steps)

    logger.info(f"num_update_steps_per_epoch {num_update_steps_per_epoch}")

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # this is for distributing
    transformer, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
            transformer, optimizer, lr_scheduler, train_dataloader
        )
    
    logger.info("Distributed model")
    
    
     # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(estimated_dataset_size / args.gradient_accumulation_steps)
    print("num_update_steps_per_epoch",num_update_steps_per_epoch)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-sd3"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {estimated_dataset_size}")
    logger.info(f"  Num batches each epoch = {total_batch_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # print trainable params
    if args.print_params:
        print_trainable_parameters(transformer)

    transformer.train()

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # prepare the inputs.
                pixel_values = batch["target_image"].to(dtype=vae.dtype)
                obj_names = batch['obj_name_image']
                prompts = batch["prompt"]

                # import pdb; pdb.set_trace()
                messages = []
                for item in obj_names:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": obj} 
                            for obj in item if isinstance(obj, Image.Image)
                        ]   
                    })

                text = []
                for idx in range(len(prompts)): 
                    obj_names_text = ", ".join(["<|vision_start|><|image_pad|><|vision_end|>" for _ in obj_names[idx]])
                    text.append("Combine the objects: " + obj_names_text + " and generate an image following: " + prompts[idx])

                # process the updated image inputs for LMM safely
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=text,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    max_length=args.num_latents,
                    truncation=True,
                    return_tensors="pt",
                )

                # process the object image inputs for Siglip
                with torch.no_grad():
                    siglip_text_inputs = siglip_processor.tokenizer(text=prompts, padding="max_length", truncation=True, return_tensors="pt")
                    siglip_image_inputs = siglip_processor(images=[obj for sublist in obj_names for obj in sublist], return_tensors="pt")

                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                # HACK unify inputs devices
                ## LMM inputs
                inputs = inputs.to(device=model_input.device, dtype=weight_dtype)
                inputs = {f"lmm_{k}": v for k, v in inputs.items()}

                ## process Siglip text inputs
                siglip_text_inputs = {k: v.to(model_input.device).to(weight_dtype) if v.is_floating_point() else v.to(model_input.device) for k, v in siglip_text_inputs.items()}
                if 'attention_mask' not in siglip_text_inputs:
                    pad_token_id = siglip_processor.tokenizer.pad_token_id or 1
                    siglip_text_inputs['attention_mask'] = (siglip_text_inputs['input_ids'] != pad_token_id).long()

                inputs['siglip_text_inputs'] = siglip_text_inputs

                ## process Siglip image inputs
                siglip_image_inputs = {
                    k: v.to(model_input.device, dtype=weight_dtype) 
                    for k, v in siglip_image_inputs.items()
                }

                inputs['siglip_image_inputs'] = siglip_image_inputs

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise


                inputs["dit_hidden_states"] = noisy_model_input
                inputs["dit_time_step"] = timesteps

                if args.random_vit_skip:
                    inputs["vit_skip_ratio"] = torch.rand(bsz, device=model_input.device, dtype=weight_dtype)

                # dit_model_pred, dit_model_pred_conn, dit_encoder_conn_proj, dit_encoder_hidden_states_proj
                if args.keep_context_embedder:
                    model_pred = transformer.forward_with_original_context_embedder(**inputs)[0]
                else:
                    loss = transformer(
                        **inputs
                    )

                # # Flow matching noise
                # # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # # Preconditioning of the model outputs.
                # if args.precondition_outputs:
                #     model_pred = model_pred * (-sigmas) + noisy_model_input

                # # these weighting schemes use a uniform timestep sampling
                # # and instead post-weight the loss
                # weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # # flow matching loss
                # if args.precondition_outputs:
                #     target = model_input
                # else:
                #     target = noise - model_input

              

                # # Compute regular loss.
                # loss = torch.mean(
                #     (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                #     1,
                # )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("connector-checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"connector-checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()



if __name__ == "__main__":
    args = parse_args()
    main(args)