import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler

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
        self.base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        # define refiner
        self.refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",text_encoder_2=self.base.text_encoder_2, vae=self.base.vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16",)
        # load it on GPUs
        self.base.to(self.device)
        self.refiner.to(self.device)
    
    def forward(self, prt):
        img = self.base(prt, 40, 0.8, "latent").images
        img_refin = self.refiner(prt, 40, 0.8, image=img).images[0]
        img_refin.save("/data/aditya/generated_image.png")



        
    