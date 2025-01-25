import torch
import numpy as np
import pandas as pd
import os
import cv2
import json

# warnings
import warnings
warnings.filterwarnings("ignore")

# in file modules
import sys
sys.path.insert(0, "/data/aditya/PRISM/")
from utils import run_gd, run_llm, utilities
from dataset.diffimages import run_sd21
# from dataset.openimages import openimages
# from utils.run_clip import retrieve_img

class pipeline4(object):
    def __init__(self, dir_pth, csv_pth):
        self.dir_pth = dir_pth
        self.csv_pth = csv_pth
    
    def forward(self):
        k=0
        for i in os.listdir(self.dir_pth):
            img_pth = os.path.join(self.dir_pth, i)
            obj = openimages(img_pth, self.csv_pth)
            # if size is not suitable
            if obj.forward()==None:
                k+=1
                print("Done",k)
                continue
            # if size is suitable
            else:
                img_rgb, segs, nouns = obj.forward()
                new_dir_pth = os.path.join("/data/aditya/visuals/",str(k+1))
                os.makedirs(new_dir_pth, exist_ok = True)
                cv2.imwrite(os.path.join(new_dir_pth,'final_img.jpg'), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                for j in range(len(segs)):
                    cv2.imwrite(os.path.join(new_dir_pth,'seg_img_'+str(nouns[j])+"_"+str(j)+'.jpg'), cv2.cvtColor(segs[j], cv2.COLOR_RGB2BGR))
                    obj1 = retrieve_img(segs[j]).retrieve_X()
                    j+=1
                k+=1
                print("Done",k)
                return obj1

class pipeline5(object):
    def __init__(self, json_pth):
        f = open(json_pth, "r")
        self.json_obj = json.load(f)
        self.llm_obj = run_llm.run_phi3()
        self.diff_obj = run_sd21()
        self.GD = run_gd.GDINO()

    def forward(self):
        for i in range(10):
            torch.cuda.empty_cache()
            prt = self.json_obj[i]["conversations"][1]['value'].split("\n\n")[0]
            if len(prt.split(" ")) < 12:
                continue
            else:
                txt = self.llm_obj.forward(prt.replace("image",""))
                sum_prt = "An image of" + txt.split("\n")[0][9:]
                nouns = txt.split("\n")[1].split(" ")[3:6]
                img_gen = run_sd21().forward(sum_prt)
                out = self.GD.predict([img_gen]*3, nouns, 0.3, 0.25,)
                for j in out:
                    temp = j['boxes'].cpu().numpy().tolist()[0]
                    out_img = utilities.visualize(img_gen, {"xmin": int(temp[0]), "ymin": int(temp[1]), "xmax": int(temp[2]), "ymax": int(temp[3])})
                    os.makedirs(os.path.join("/data/aditya/visuals1/",str(i)), exist_ok =True)
                    out_img.save(os.path.join(os.path.join("/data/aditya/visuals1/",str(i)),"output_image_"+j["labels"][0]+".png"))
                    
if __name__ == "__main__":
    # for pipeline 4
    # dir_pth = "/data/datasets/openimages/images/" 
    # csv_pth = "/data/datasets/openimages/small_op_annotations.csv"
    # new_algorithms(dir_pth, csv_pth).pipeline4()

    # for pipeline 5
    json_pth = "/data/datasets/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k.json"
    pipeline5(json_pth).forward()


