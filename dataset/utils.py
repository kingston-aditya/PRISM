import math
import numpy as np
import json

def correct_inputs(imgs, txts):
    temp = {}
    for i in range(len(txts)):
        for j in txts[i].split(","):
            temp[j] = imgs[i]
    return temp

def pretty_output(bbox_lst, filname_lst, noun_lst, cap_lst, f):
    k = 0
    for i in range(len(noun_lst)):
        object_temp = bbox_lst[k:k+len(noun_lst[i].split(","))]
        atema = []
        for item in object_temp:
            if len(item['scores'].cpu().tolist()) !=0:
                xmin = math.ceil(np.asarray(item["boxes"].to("cpu"))[0][0])
                ymin = math.ceil(np.asarray(item["boxes"].to("cpu"))[0][1])
                xmax = math.ceil(np.asarray(item["boxes"].to("cpu"))[0][2])
                ymax = math.ceil(np.asarray(item["boxes"].to("cpu"))[0][3])
                labels = item["labels"][0]
                filname = str(filname_lst[i])
                atema.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels, "img_pth": filname})
            else:
                pass
        temp = {"file_name": filname_lst[i], "prompt": cap_lst[i], "object": atema}
        k += len(noun_lst[i].split(","))
        f.write(json.dumps(temp) + '\n')

def dynamic_collate(batch):
    it = [item for item in batch]
    return it

def dynamic_collate_1(batch):
    it_nouns = [item["nouns"] for item in batch]
    it_imgs = [item["images"] for item in batch]

    return {
        "nouns": it_nouns,
        "images": it_imgs
    }