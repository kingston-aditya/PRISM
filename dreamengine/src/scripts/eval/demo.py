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

import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/PRISM/dreamengine/src/diffusers/src/")
from diffusers.models.transformers.transformer_sd3 import (
    SD3Transformer2DModel,
    QwenVLSD3_DirectMap_Transformer2DModel as QwenVLSD3Transformer2DModel
)
from diffusers.pipelines.stable_diffusion_3.pipeline_qwen_vl_stable_diffusion_3 import QwenVLStableDiffusion3Pipeline
from diffusers import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig

import os

import pdb

# set CUDA_VISIBLE_DEVICES to 0
# Params
MODEL_PATH = "/nfshomes/asarkar6/trinity/model_weights/"
QWEN_PATH = "Qwen/Qwen2-VL-2B-Instruct"
SD3_PATH = "stabilityai/stable-diffusion-3.5-large"
VALID_PATH = "/nfshomes/asarkar6/aditya/PRISM/validation"
OUTPUT_PATH = "/nfshomes/asarkar6/aditya/gen_images/"
DreamEngine_CKPT_DIR= f"{MODEL_PATH}/DreamEngine-ObjectFusion" # https://huggingface.co/leonardPKU/DreamEngine-ObjectFusion

# Model initialization
qwenvl2_model = Qwen2VLForConditionalGeneration.from_pretrained(QWEN_PATH, cache_dir=MODEL_PATH, device_map="cuda", torch_dtype=torch.bfloat16)
sd3_model = SD3Transformer2DModel.from_pretrained(SD3_PATH, subfolder="transformer", cache_dir=MODEL_PATH, torch_dtype=torch.bfloat16)
sd3_model = sd3_model.to("cuda")

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
qwenvl2_model.add_adapter(lmm_lora_config)
sd3_model.add_adapter(transformer_lora_config)

def load_sharded_model(config_path, qwenvl2_model, sd3_model, index_path, bin_files_folder, device='cpu',dtype=torch.bfloat16):
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
    
    del qwenvl2_model, sd3_model  

    # Transfer the model to the specified device
    print(f"Transferring the model to {device.upper()}...")
    model.to(dtype=dtype).to(device)  # First change dtype, then device
    model.eval()  # Set the model to evaluation mode

    print("Model loaded successfully.")
    return model


model = load_sharded_model(
    config_path=DreamEngine_CKPT_DIR+"/transformer/config.json",
    qwenvl2_model=qwenvl2_model,
    sd3_model=sd3_model,
    index_path=DreamEngine_CKPT_DIR+"/transformer/diffusion_pytorch_model.bin.index.json",
    bin_files_folder=DreamEngine_CKPT_DIR+"/transformer",
    device='cuda',
    dtype=torch.bfloat16
)

# Initialize other components
noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    SD3_PATH, subfolder="scheduler", cache_dir=MODEL_PATH, 
)
processor = AutoProcessor.from_pretrained(QWEN_PATH, max_pixels=512*28*28)
vae = AutoencoderKL.from_pretrained(
    SD3_PATH,
    subfolder="vae",
    cache_dir=MODEL_PATH,
).to("cuda", dtype=torch.bfloat16)

pipeline = QwenVLStableDiffusion3Pipeline(
    model,
    processor,
    noise_scheduler,
    vae
)

obj_transform = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    ]
)

def generate_image(image1, image2, caption1, caption2, text_prompt, cfg_scale, size, num_steps, seed):
    """Generate an image based on input parameters."""
    
    torch.manual_seed(seed)

    segments = [caption1,obj_transform(image1)],[caption2,obj_transform(image2)]

    output = pipeline.cfg_predict(prompt=text_prompt,segments=None,num_inference_steps=num_steps,num_images_per_prompt=1,width=size,height=size, guidance_scale=cfg_scale, max_sequence_length=334)

    return output[0][0]    

if __name__ == "__main__":
    # validation examples
    f = open(os.path.join(VALID_PATH, "metadata.jsonl"), "r")
    json_obj = {"image":[], "prompt":[], "object":[]}
    for idx, line in enumerate(f):
        temp = json.loads(line.strip())
        prompt = temp["prompt"]
        labels_0 = temp["object"][0]["labels"]
        labels_1 = temp["object"][1]["labels"]
        image_0 = os.path.join(VALID_PATH, temp["object"][0]["img_pth"])
        image_1 = os.path.join(VALID_PATH, temp["object"][1]["img_pth"])

        # generate the images
        output = generate_image(Image.open(image_0),
            Image.open(image_1),
            labels_0,
            labels_1,
            prompt,
            3.5,
            768,
            28,
            136147
        ) 
        output.save(os.path.join(OUTPUT_PATH, f"output_{idx}.png"))

        torch.cuda.empty_cache()





