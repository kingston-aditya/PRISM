import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from PIL import Image

def read_object_image(img_pth, bbx):
    # open image
    img = Image.open(img_pth)
    img_array = np.array(img)

    # get the objects 
    c = {}
    for i in range(len(bbx)):
        obj_img_array = img_array[int(bbx[i]["ymin"]):int(bbx[i]["ymax"]), int(bbx[i]["xmin"]):int(bbx[i]["xmax"])]
        obj_img = obj_img_array
        c[bbx[i]["labels"]] = obj_img
    return c, img_array

if __name__ == "__main__":
    f = open("/nfshomes/asarkar6/trinity/trinity-data-real2.json", "r")
    js = json.load(f)
    f.close()

    x = {}; y={}
    for i in range(len(js)):
        print(i)
        if js[i]['bbox'] is not None:
            c, img = read_object_image(js[i]['img_pth'], js[i]['bbox'])
            x[i] = {'caps': js[i]['caps'], 'objs': c}
            y[i] = img

    np.save('/nfshomes/asarkar6/trinity/trinity_x1.npy', list(x.values()))
    np.save('/nfshomes/asarkar6/trinity/trinity_y1.npy', list(y.values()))

