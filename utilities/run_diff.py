import os
import torch
from torchvision import transforms

# from huggingface_hub import login
# login(token = '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')
from PIL import Image

from diffusers import FluxPipeline, DiffusionPipeline 
from accelerate import PartialState
from accelerate.utils import gather_object

# flux model pipeline
class run_flux(object):
    def __init__(self, args):
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", token="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",  cache_dir = args["cache_dir"], torch_dtype=torch.bfloat16)
        self.pipe.set_progress_bar_config(disable=True)
        # load it on multiple GPUs
        self.distributed_state = PartialState()
        self.pipe.to(self.distributed_state.device)

    def forward(self, prt):
        with self.distributed_state.split_between_processes(prt) as prompts:
            img = self.pipe(prompt = prompts, height=512, width=512, num_inference_steps=50).images
        
        self.distributed_state.wait_for_everyone()
        img = gather_object(img)

        return img

# diffusion model pipeline
class run_sdxl(object):
    def __init__(self, args):
        # define base
        self.base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", cache_dir = args["cache_dir"], torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        self.refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", text_encoder_2=self.base.text_encoder_2, vae=self.base.vae, torch_dtype=torch.float16, cache_dir = args["cache_dir"], use_safetensors=True, variant="fp16",)
        # remove inference bars
        self.base.set_progress_bar_config(disable=True)
        self.refiner.set_progress_bar_config(disable=True)
        # load it on multiple GPUs
        self.distributed_state = PartialState()
        self.base.to(self.distributed_state.device)
        self.refiner.to(self.distributed_state.device)
    
    def forward(self, prt):
        with self.distributed_state.split_between_processes(prt) as prompts:
            img = self.base(prompt = prompts, num_inference_steps=50, output_type = "latent").images
            img = self.refiner(prompt = prompts, num_inference_steps = 50, image=img).images
        
        self.distributed_state.wait_for_everyone()
        img = gather_object(img)

        return img

if __name__ == "__main__":
    args = {"cache_dir": "/nfshomes/asarkar6/trinity/model_weights/"}
    prts1 = [
        "A serene beach at sunset with soft waves, pastel skies, and gentle clouds.",
        "A futuristic city skyline at dusk, with glowing neon lights and towering skyscrapers.",
        "A peaceful forest with tall trees, misty atmosphere, and rays of sunlight filtering through the leaves.",
        "A rolling grassy meadow under a bright blue sky with a few fluffy clouds.",
        "A starry night sky with a galaxy in the distance, glowing nebulae, and a distant planet.",
    ]
    prts2 = [
        "A cozy winter scene with snow-covered trees, soft snowfall, and a warm glow in the distance.",
        "A vast desert landscape with golden sand dunes stretching to the horizon, under a clear sky.",
        "A calm mountain lake surrounded by dense pine forests, with clear blue skies and reflection of mountains.",
        "A dreamy cloudscape with cotton candy clouds in pastel pinks, purples, and blues.",
        "A lush tropical jungle with vibrant green foliage, exotic flowers, and a waterfall in the background."
    ]
    img1 = run_sdxl(args).forward(prts1)
    img2 = run_sdxl(args).forward(prts2)

    idx = 0
    for image in img1:
        image.save("/nfshomes/asarkar6/aditya/PRISM/backgrounds/"+str(idx)+".jpg")
        idx+=1

    for image in img2:
        image.save("/nfshomes/asarkar6/aditya/PRISM/backgrounds/"+str(idx)+".jpg")
        idx+=1

    







        
    