import torch
import torch.nn as nn
from transformers import CLIPProcessor, 
from utils import Transformer, ProjectionBlock
import sys
sys.path.insert(1, "/nfshomes/asarkar6/PRISM/diffusers")
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

def find_idx(caps, nouns):
    caps = (caps/torch.norm(caps, dim=1, keepdim=True)).to("cuda")
    idx = []
    for i in nouns:
        i = (i/torch.norm(i)).to("cuda")
        # output of size 1xN
        res = torch.matmul(caps, i.T)
        idx.append(torch.argmax(res))
    return idx

class pipeline1(StableDiffusionXLPipeline):
    def __init__( 
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super(StableDiffusionXLPipeline, self).__init__(vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, scheduler)
        self.transformer = Transformer(77, 48, 7, 0.6, 0.1)
        self.proj_layer = ProjectionBlock(512, 0.6, 77, 0.1)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


    def encode_prompt(self, prompt, clip_obj):
        device = "cuda"

        # find the indices
        caps_inputs = self.tokenizer(prompt["caps"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",)
        noun_inputs = [self.tokenizer(i, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",) for i in prompt["nouns"]]
        idx = find_idx(caps_inputs, noun_inputs)

        # get image embeddings
        img_inputs = []
        for i in prompt["bb"]:
            with torch.no_grad():
                inputs = self.processor(images=i, return_tensors="pt")
                image_embeddings = self.clip_model.get_image_features(inputs["pixel_values"])
                image_embeddings = image_embeddings/image_embeddings.norm(p=2, dim=-1, keepdim=True)
            img_inputs.append(image_embeddings)
        img_inputs = torch.stack([self.proj_layer(i) for i in img_inputs])

        # compute the multimodal prompt embeds
        caps_input_ids = caps_inputs.input_ids.to(device)
        img_inputs = img_inputs.to(device)
        prompt_embeds = self.transformer(caps_input_ids, img_inputs, idx)
        
        return prompt_embeds








        


