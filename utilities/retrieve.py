from run_vlm import clip_embeds, load_img, trans_form
import numpy as np

class retrieve_img(object):
    def __init__(self, img_rgb):
        self.img_query = clip_embeds().forward_img(load_img(img_rgb))
        self.device = "cuda"
        Xr = np.load("/data/aditya/coco_embeds/coco_img_feat_0.npy")
        self.Xr = Xr/np.linalg.norm(Xr, axis=-1, keepdims=True)
        self.Xr = torch.from_numpy(self.Xr).float().to(self.device)

    def retrieve_X(self):
        q = torch.from_numpy(self.img_query.T).float().to(self.device)
        ans = torch.matmul(self.Xr,q)
        ind = np.argmax(ans.cpu().detach().numpy())
        return ind