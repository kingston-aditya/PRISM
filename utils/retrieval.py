import numpy as np
import torch
from PIL import Image
from clip_embeds import make_embeds
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

class retrieve_img(object):
    def __init__(self, img_rgb):
        self.img_query = make_embeds().forward_img(load_img(img_rgb))
        self.device = "cuda"
        Xr = np.load("/data/aditya/coco_embeds/coco_img_feat_0.npy")
        self.Xr = Xr/np.linalg.norm(Xr, axis=-1, keepdims=True)
        self.Xr = torch.from_numpy(self.Xr).float().to(self.device)

    def retrieve_X(self):
        q = torch.from_numpy(self.img_query.T).float().to(self.device)
        ans = torch.matmul(self.Xr,q)
        ind = np.argmax(ans.cpu().detach().numpy())
        return ind