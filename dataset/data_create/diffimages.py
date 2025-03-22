import torch
import os
from huggingface_hub import login
login(token = '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')
from diffusers import FluxPipeline, DiffusionPipeline, StableDiffusion3Pipeline

class run_sd21(object):
    def __init__(self):
        self.device = "cuda"
        self.pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
    
    def forward(self, prt):
        img = self.pipe(prt).images[0]
        return img

class run_sdxl(object):
    def __init__(self):
        self.device = "cuda"
        # define base
        self.base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", cache_dir = "/nfshomes/asarkar6/trinity/model_weights/", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        # define refiner
        self.refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",text_encoder_2=self.base.text_encoder_2, vae=self.base.vae, torch_dtype=torch.float16, cache_dir = "/nfshomes/asarkar6/trinity/model_weights/", use_safetensors=True, variant="fp16",)
        # load it on GPUs
        self.base.to(self.device)
        self.refiner.to(self.device)
    
    def forward(self, prt):
        img = self.base(prompt = prt, num_inference_steps = 50, output_type = "latent").images
        img_refin = self.refiner(prompt = prt, num_inference_steps = 50, image=img).images[0]
        return img_refin

class run_flux(object):
    def __init__(self):
        self.device = "cuda"
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", token="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",  cache_dir = "/nfshomes/asarkar6/trinity/model_weights/", torch_dtype=torch.bfloat16)
        self.pipe.to(self.device)

    def forward(self, prt):
        img = self.pipe(prt, height=768, width=768, num_inference_steps=50).images[0]
        return img

if __name__ == "__main__":
    img = run_flux().forward("A majestic lion jumping from a big stone at night.")
    img.save("/nfshomes/asarkar6/PRISM/generated_image.png")







        
    