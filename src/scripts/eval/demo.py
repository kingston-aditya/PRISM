# Build a gradio demo for text-image interleaved image generation
# Input: two images and their corresponding captions, plus a text instruction
# For example:
# Input:
# Image 1: A dog image
# Image 1 caption: A dog
# Image 2: A cat image
# Image 2 caption: A cat
# Text instruction: Combine the two animals into one animal
# Output: A synthesized animal image

# The model also accepts some parameters:
# cfg_scale: A scalar to control the quality of generated image
# size: From 512x512 to 1024x1024
# num_steps: 28
# seed: A scalar to control the randomness of generated image

# Temporarily hardcode the generation function, input is image1, image2, caption1, caption2, text_prompt, cfg_scale, size, num_steps, seed
# Output is the generated image

# Consolidated imports
import os
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from pathlib import Path

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import logging

import sys
sys.path.insert(1, "/home/saividyaranya/PRISM/src/diffusers/src")
from diffusers.models.transformers.transformer_sd3 import (
    SD3Transformer2DModel,
    QwenVLSD3_DirectMap_Transformer2DModel as QwenVLSD3Transformer2DModel
)
from diffusers.pipelines.stable_diffusion_3.pipeline_qwen_vl_stable_diffusion_3 import QwenVLStableDiffusion3Pipeline
from diffusers import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from PIL import Image
from torchvision import transforms
from peft import LoraConfig
import glob, json
from tqdm.auto import tqdm

import sys
import pdb as pdb_original
from torch.utils.data import Dataset
from transformers import AutoConfig

from huggingface_hub import login
login(token="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")

# set CUDA_VISIBLE_DEVICES to 0
# Constants
MODEL_PATH = "/data/home/saividyaranya/PRISM/model_weights3/dream_engine/"
QWEN_PATH = "Qwen/Qwen2-VL-2B-Instruct"
SD3_PATH = "stabilityai/stable-diffusion-3.5-large"
DreamEngine_CKPT_DIR= f"{MODEL_PATH}" # https://huggingface.co/leonardPKU/DreamEngine-ObjectFusion
dataset_path = "/data/home/saividyaranya/PRISM/validation/"
output_dir = "/home/saividyaranya/PRISM/all_output_logs/dreamengine_images"

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

class InferDataset(Dataset):
    def __init__(self, temp, dataset_pth):
        # load dataset
        self.temp = temp
        self.dataset_pth = dataset_pth
    
    def __getitem__(self, idx):        
        # get the image objects
        bbox_values = []
        bbox_labs = []
        bbox_info = self.temp["object"][idx]
        if len(bbox_info) > 0:
            # get the prompt tokens
            with torch.no_grad():
                prompt_toks = self.temp["prompt"][idx]

            # process the bbox
            for idx, item in enumerate(bbox_info):
                img_mat = Image.open(os.path.join(self.dataset_pth, item["img_pth"])).convert("RGB")
                bbox_values.append(img_mat)
                bbox_labs.append(item["labels"])

        elif len(bbox_info) == 0:
            raise Exception("Give me an object.")
            
        return {
            "prompt_embeds": prompt_toks,
            "object_prompt_embeds": bbox_values,
            "object_labels": bbox_labs,
        }

    def __len__(self):
        return len(self.temp["prompt"])

def generate_image(obj_images, obj_labels, text_prompt, pipeline, obj_transform):
    """Generate an image based on input parameters."""
    torch.manual_seed(42)

    segments = []
    for idx, item in enumerate(obj_images[0]):
        
        segments.append(["An image of "+obj_labels[0][idx], obj_transform(item)])

    segments = tuple(segments)
    ForkedPdb().set_trace()
    output = pipeline.cfg_predict(prompt=text_prompt[0], segments=segments, num_inference_steps=28, num_images_per_prompt=4,width=1024,height=1024, guidance_scale=3.5, max_sequence_length=334)

    return output

# reads a batch of image-text pairs  
def read_eval_dataset():
    # read multiple files
    json_obj = {"image":[], "prompt":[], "object":[]}

    for name in glob.glob(os.path.join(dataset_path, "*.jsonl")):
        with open(os.path.join(dataset_path, name), "r") as f:
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

def main():
    logging_dir = Path(output_dir, os.path.join(output_dir, "logs"))

    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Model initialization - SD3
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        SD3_PATH, subfolder="scheduler"
    )

    processor = AutoProcessor.from_pretrained(QWEN_PATH, max_pixels=512*28*28)
    vae = AutoencoderKL.from_pretrained(
        SD3_PATH,
        subfolder="vae",
        cache_dir=MODEL_PATH
    ).to(accelerator.device, dtype=torch.bfloat16)


    sd3_model = SD3Transformer2DModel.from_pretrained(SD3_PATH, subfolder="transformer", cache_dir=MODEL_PATH).to(accelerator.device)

    transformer_lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        init_lora_weights="gaussian",
        target_modules=[
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
            "attn.to_add_out", "attn.to_k", "attn.to_out.0",
            "attn.to_q", "attn.to_v",
        ]
    )

    # Apply LoRA configurations
    sd3_model.add_adapter(transformer_lora_config)
    sd3_model.to(accelerator.device)

    # Model initialization - Qwen 2
    qwenvl2_model = Qwen2VLForConditionalGeneration.from_pretrained(QWEN_PATH, cache_dir=MODEL_PATH)
    qwenvl2_model.to(accelerator.device)

    # LoRA configurations
    lmm_lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        init_lora_weights="gaussian",
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]
    )
    qwenvl2_model.add_adapter(lmm_lora_config)

    def load_sharded_model(config_path, index_path, bin_files_folder, device=accelerator.device,dtype=torch.bfloat16):
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
        model = QwenVLSD3Transformer2DModel(qwenvl2_model, sd3_model)

        del qwenvl2_model, sd3_model

        model.to(accelerator.device)
        
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
            bin_state = torch.load(bin_path, map_location="cpu")
            
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
        print(f"Transferring the model to {device}...")
        model.to(dtype=dtype).to(device)  # First change dtype, then device
        model.eval()  # Set the model to evaluation mode

        print("Model loaded successfully.")
        return model

    model = load_sharded_model(
        config_path=DreamEngine_CKPT_DIR+"/transformer/config.json",
        index_path=DreamEngine_CKPT_DIR+"/transformer/diffusion_pytorch_model.bin.index.json",
        bin_files_folder=DreamEngine_CKPT_DIR+"/transformer",
        device=accelerator.device,
        dtype=torch.bfloat16
    )

    pipeline = QwenVLStableDiffusion3Pipeline(
        model,
        processor,
        noise_scheduler,
        vae
    )
    pipeline.to(accelerator.device)

    del model, vae

    pipeline = accelerator.prepare(pipeline)

    obj_transform = transforms.Compose(
        [
            transforms.Resize(336, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
    )

    # load the dataset
    json_obj = read_eval_dataset()

    def collate_fn(batch):
        prompt_embeds = [item["prompt_embeds"] for item in batch]
        object_prompt_embeds = [item["object_prompt_embeds"] for item in batch]
        object_labels = [item["object_labels"] for item in batch]
        return {
            "prompt_embeds": prompt_embeds,
            "object_prompt_embeds": object_prompt_embeds,
            "object_labels": object_labels,
        }

    precomputed_dataset = InferDataset(json_obj, dataset_path)

    # author's code can handle only 1 image at once.
    eval_dataloader = torch.utils.data.DataLoader(
        precomputed_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=4,
    )

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Inferring")):
        
        images = generate_image(batch["object_prompt_embeds"], batch["object_labels"], batch["prompt_embeds"], pipeline, obj_transform)
        for idx, item in enumerate(images):
            ForkedPdb().set_trace()
            item.save(os.path.join(output_dir, f"prompt{step}_img{idx}.png"))


if __name__ == "__main__":
    main()





