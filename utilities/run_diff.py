import torch
from huggingface_hub import login
login(token = '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')
from diffusers import FluxPipeline, DiffusionPipeline

class run_sdxl(object):
    def __init__(self, args):
        self.device = "cuda"
        # define base
        self.base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", cache_dir = args.cache_dir, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        self.refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",text_encoder_2=self.base.text_encoder_2, vae=self.base.vae, torch_dtype=torch.float16, cache_dir = args.cache_dir, use_safetensors=True, variant="fp16",)
        # remove inference bars
        self.base.set_progress_bar_config(disable=True)
        self.refiner.set_progress_bar_config(disable=True)
        # load it on GPUs
        self.base.to(self.device)
        self.refiner.to(self.device)
    
    def forward(self, prt):
        img = self.base(prompt = prt, num_inference_steps = 50, output_type = "latent").images
        img_refin = self.refiner(prompt = prt, num_inference_steps = 50, image=img).images
        return img_refin

class run_flux(object):
    def __init__(self, args):
        self.device = "cuda"
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", token="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",  cache_dir = args.cache_dir, torch_dtype=torch.bfloat16)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(self.device)

    def forward(self, prt):
        img = self.pipe(prompt = prt, height=512, width=512, num_inference_steps=50).images
        return img

if __name__ == "__main__":
    prts = ["A majestic lion jumping from a big stone at night.", "A majestic cat jumping from a big stone at night.", "A majestic cat jumping from a big stone in morning."]
    img = run_sdxl().forward(prts)
    # img.save("/nfshomes/asarkar6/PRISM/generated_image.png")







        
    