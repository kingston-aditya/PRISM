import torch
import json
import glob
import math
import os
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(1, "/nfshomes/asarkar6/aditya/PRISM/qwen-image")
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
from diffusers import AutoencoderKLQwenImage, QwenImageEditPipeline, QwenImageTransformer2DModel
from diffusers.image_processor import VaeImageProcessor

import pdb as pdb_original


OUTPUT_DIR = "/nfshomes/asarkar6/aditya/gen_images/"
INPUT_DIR = "/nfshomes/asarkar6/aditya/PRISM/validation"
CACHE_DIR = "/nfshomes/asarkar6/trinity/model_weights/"

SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  
Please strictly follow the rewriting rules below:
## 1. General Principles
- Keep the rewritten prompt **concise**. Avoid overly long sentences and reduce unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the edited input image's overall scene.  
## 2. Task Type Handling Rules
### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
    > Original: "Add an animal"  
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"  
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  
### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes " ". Do not translate or alter the original language of the text, and do not change the capitalization.  
- **For text replacement tasks, always use the fixed template:**
    - Replace "xx" to "yy".  
    - Replace the xx bounding box to "yy".  
- If the user does not specify text content, infer and add concise text based on the instruction and the input image's context. For example:  
    > Original: "Add a line of text" (poster)  
    > Rewritten: "Add text "LIMITED EDITION" at the top center with slight shadow"  
- Specify text position, color, and layout in a concise way.  
### 3. Human Editing Tasks
- Maintain the person's core visual consistency (ethnicity, gender, age, hairstyle, expression, outfit, etc.).  
- If modifying appearance (e.g., clothes, hairstyle), ensure the new element is consistent with the original style.  
- **For expression changes, they must be natural and subtle, never exaggerated.**  
- If deletion is not specifically emphasized, the most important subject in the original image (e.g., a person, an animal) should be preserved.
    - For background change tasks, emphasize maintaining subject consistency at first.  
- Example:  
    > Original: "Change the person's hat"  
    > Rewritten: "Replace the man's hat with a dark brown beret; keep smile, short hair, and gray jacket unchanged"  
### 4. Style Transformation or Enhancement Tasks
- If a style is specified, describe it concisely with key visual traits. For example:  
    > Original: "Disco style"  
    > Rewritten: "1970s disco: flashing lights, disco ball, mirrored walls, colorful tones"  
- If the instruction says "use reference style" or "keep current style," analyze the input image, extract main features (color, composition, texture, lighting, art style), and integrate them concisely.  
- **For coloring tasks, including restoring old photos, always use the fixed template:** "Restore old photograph, remove scratches, reduce noise, enhance details, high resolution, realistic, natural skin tones, clear facial features, no distortion, vintage photo restoration"  
- If there are other changes, place the style description at the end.
## 3. Rationality and Logic Checks
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" should be logically corrected.  
- Add missing key information: if position is unspecified, choose a reasonable area based on composition (near subject, empty space, center/edges).  
# Output Format
Return only the rewritten instruction text directly, without JSON formatting or any other wrapper.
'''

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

class QwenImagePromptEncoder:
    def __init__(self, pretrained_model_name_or_path, device, cache_dir, torch_dtype = torch.bfloat16):
        self.encoder_ = QwenImageEditPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            scheduler=None,
            transformer=None,
            vae=None,
            cache_dir=cache_dir
        ).to(device)

    def __call__(self, prompt, image, **kwargs):
        inputs = {
            "image": image,
            "prompt": prompt
        }
        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = self.encoder_.encode_prompt(**inputs)
            return prompt_embeds, prompt_embeds_mask


class QwenImageDecoder:
    """
    Based off QwenImagePipeline
    """
    def __init__(self, pretrained_model_name_or_path, device, cache_dir, torch_dtype = torch.bfloat16):
        self.vae_ = AutoencoderKLQwenImage.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            subfolder="vae",
            cache_dir=cache_dir,
        ).to(device)

        self.vae_scale_factor_ = 2 ** len(self.vae_.temperal_downsample)
        self.image_processor_ = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor_ * 2)

    @property
    def vae_scale_factor(self):
        return self.vae_scale_factor_

    def __call__(self, latents: torch.Tensor, output_type: str = "pil"):
        vae = self.vae_
        latents = latents.to(vae.device)

        with torch.no_grad():
            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = vae.decode(latents, return_dict=False)[0][:, :, 0]

            image = self.image_processor_.postprocess(image, output_type=output_type)
            return image


class QwenImageTransformer:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        vae_scale_factor: float,
        device: str,
        cache_dir: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        transformer = QwenImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            subfolder="transformer",
            cache_dir=cache_dir
        ).to(device)

        self.vae_ = AutoencoderKLQwenImage.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            subfolder="vae",
            cache_dir=cache_dir
        ).to(device)

        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }

        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        transformer_pipeline = QwenImageEditPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
            vae=self.vae_,
            # scheduler=scheduler,
            cache_dir=cache_dir,
        ).to(device)
        transformer_pipeline.transformer.__class__ = QwenImageTransformer2DModel
        # transformer_pipeline.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

        self.transformer_pipeline_ = transformer_pipeline

        self.vae_scale_factor_ = vae_scale_factor
        self.device_ = device

        self.transformer_pipeline_.set_progress_bar_config(disable=None)

    @property
    def device(self):
        return self.device_

    def __call__(self, height: int, width: int, **kwargs) -> torch.Tensor:
        # ForkedPdb().set_trace()
        latents = self.transformer_pipeline_(output_type="latent", height=height, width=width, **kwargs)
        latents = self.transformer_pipeline_._unpack_latents(latents[0], height, width, self.vae_scale_factor_)
        return latents
    
# concatenate a list of PIL images
def concatenate_images(images, direction="horizontal"):
    if not images:
        return None
    
    # Filter out None images
    valid_images = [Image.open(os.path.join(INPUT_DIR, img["img_pth"])) if img is not None else None for img in images ]
    
    if not valid_images:
        return None
    
    if len(valid_images) == 1:
        return valid_images[0].convert("RGB")
    
    # Convert all images to RGB
    valid_images = [img.convert("RGB") for img in valid_images]
    
    if direction == "horizontal":
        # Calculate total width and max height
        total_width = sum(img.width for img in valid_images)
        max_height = max(img.height for img in valid_images)
        
        # Create new image
        concatenated = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        
        # Paste images
        x_offset = 0
        for img in valid_images:
            # Center image vertically if heights differ
            y_offset = (max_height - img.height) // 2
            concatenated.paste(img, (x_offset, y_offset))
            x_offset += img.width
            
    else:  
        # Calculate max width and total height
        max_width = max(img.width for img in valid_images)
        total_height = sum(img.height for img in valid_images)
        
        # Create new image
        concatenated = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        
        # Paste images
        y_offset = 0
        for img in valid_images:
            # Center image horizontally if widths differ
            x_offset = (max_width - img.width) // 2
            concatenated.paste(img, (x_offset, y_offset))
            y_offset += img.height
    
    return concatenated

def polish_prompt(result):
    if '{"Rewritten"' in result:
        try:
            # Clean up the response
            result = result.replace('```json', '').replace('```', '')
            result_json = json.loads(result)
            polished_prompt = result_json.get('Rewritten', result)
        except:
            polished_prompt = result
    else:
        polished_prompt = result
    return polished_prompt

def transform_image(images):
    resize_transform = transforms.Resize((768, 512))
    return resize_transform(images)


def main1(encoder, decoder, transformer_pipeline, prompt, images, negative_prompt=" "):
    # brief preprocessing
    images = concatenate_images(images, direction="horizontal")
    images = transform_image(images)

    new_prompt = polish_prompt(f"{SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nRewritten Prompt:")

    # encode the prompts
    positive_prompt_embeds, positive_prompt_embeds_mask = encoder(prompt=new_prompt, image=images)
    negative_prompt_embeds, negative_prompt_embeds_mask = encoder(prompt=negative_prompt, image=images)

    # ForkedPdb().set_trace()

    # generate the image latents
    latents = transformer_pipeline(
        image=images.convert("RGB"),
        prompt_embeds=positive_prompt_embeds.to(transformer.device),
        prompt_embeds_mask=positive_prompt_embeds_mask.to(transformer.device),
        negative_prompt_embeds=negative_prompt_embeds.to(transformer.device),
        negative_prompt_embeds_mask=negative_prompt_embeds_mask.to(transformer.device),
        num_inference_steps=30,
        height=512,
        width=512,
    )

    # decode the image from the latent
    output_image = decoder(latents)

    return output_image

if __name__ == "__main__":
    pretrained_model_name_or_path = "Qwen/Qwen-Image-Edit"
    torch_dtype = torch.bfloat16

    # load the dataset
    json_obj = {"prompt":[], "object":[]}
    for name in glob.glob(os.path.join(INPUT_DIR, "*.jsonl")):
        with open(os.path.join(INPUT_DIR, name), "r") as f:
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

    # load the pipelines
    encoder = QwenImagePromptEncoder(pretrained_model_name_or_path, "cuda:0", CACHE_DIR, torch_dtype)
    decoder = QwenImageDecoder(pretrained_model_name_or_path, "cuda:0", CACHE_DIR, torch_dtype)
    transformer = QwenImageTransformer(pretrained_model_name_or_path, decoder.vae_scale_factor, "cuda:1", CACHE_DIR, torch_dtype)

    # iterate over the whole dataset
    for idx, (prompt_item, object_item) in enumerate(zip(json_obj["prompt"], json_obj["object"])):
        output_img = main1(encoder, decoder, transformer, prompt_item, object_item)
        output_img[0].save(os.path.join(OUTPUT_DIR, f"qwen_image_edit_{idx}.jpg"))