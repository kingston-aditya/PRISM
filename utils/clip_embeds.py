import torch
from dassl.clip import clip
from dassl.coop import load_clip_to_cpu_name

class make_embeds(object):
    def __init__(self, backname="ViT-B/16"):
        super(make_embeds, self).__init__()
        self.device = "cuda"
        clip_model = load_clip_to_cpu_name(backname)
        clip_model.to(self.device)
        self.clip_model = clip_model
    
    def forward_img(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().detach().numpy()

