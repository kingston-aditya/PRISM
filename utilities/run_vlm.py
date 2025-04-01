import numpy as np
import torch
from PIL import Image
from dassl.clip import clip
from dassl.coop import load_clip_to_cpu_name
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def trans_form(n_px=224):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def load_img(img_rgb):
    obj = trans_form(224)
    img_enc = obj(Image.fromarray(img_rgb.astype(np.uint8))).to("cuda")
    img_res = torch.reshape(img_enc,(1,3,224,224))
    return img_res

class clip_embeds(object):
    def __init__(self, backname="ViT-B/16"):
        super(clip_embeds, self).__init__()
        self.device = "cuda"
        clip_model = load_clip_to_cpu_name(backname)
        clip_model.to(self.device)
        self.clip_model = clip_model
    
    def forward_img(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().detach().numpy()

