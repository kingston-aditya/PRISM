import torch
import numpy as np
import pandas as pd
import os
import cv2
import json
from PIL import Image

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
        # self.GD = run_gd.GDINO()

    def forward(self):
        cn = {}
        for i in range(20):
            torch.cuda.empty_cache()
            prt = self.json_obj[i]["conversations"][1]['value'].split("\n\n")[0]
            if len(prt.split(" ")) < 12:
                continue
            else:
                self.llm_obj = run_llm.run_phi3()
                txt = self.llm_obj.forward(prt.replace("image",""))
                sum_prt = "An image of" + txt.split("\n")[0][9:]
                nouns = txt.split("\n")[1].split(" ")[3:]
                print("NOUNS", nouns)
                del self.llm_obj
                self.diff_obj = run_sd21()

                # save an image
                img_gen = self.diff_obj.forward(sum_prt)
                img_gen.save(os.path.join("/data/aditya/visuals1/","output_image_"+str(i)+".png"))

                # write a caption
                f = open(os.path.join("/data/aditya/visuals1/","output_image_"+str(i)+".txt"), "w")
                f.write(sum_prt)
                f.close()

                del self.diff_obj
                cn["output_image_"+str(i)+".png"] = nouns
        torch.cuda.empty_cache()

        self.GD = run_gd.GDINO()
        k = 0
        # [list(cn.keys())]
        for i in list(cn.keys()):
            print("Done", k)
            img_gen = Image.open(os.path.join("/data/aditya/visuals1/",i))
            out = self.GD.predict([img_gen]*len(cn[i]), cn[i], 0.3, 0.25,)
            # print(out)
            out1 = [{"labels": i["labels"][0], "boxes":i['boxes'].cpu().numpy().tolist()[0]} for i in out if len(i['boxes'].cpu().numpy().tolist()[0]) != 0]
            out_fil = utilities.find_important(out1, img_gen.size)
            for j in out_fil:
                out_img = utilities.visualize(img_gen, j)
                os.makedirs(os.path.join("/data/aditya/visuals1/",str(k)), exist_ok =True)
                out_img.save(os.path.join(os.path.join("/data/aditya/visuals1/",str(k)),"output_image_"+j["labels"]+".png"))
            k+=1

if __name__ == "__main__":
    # for pipeline 4
    # dir_pth = "/data/datasets/openimages/images/" 
    # csv_pth = "/data/datasets/openimages/small_op_annotations.csv"
    # new_algorithms(dir_pth, csv_pth).pipeline4()

    # for pipeline 5
    json_pth = "/data/datasets/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k.json"
    pipeline5(json_pth).forward()


