import torch
import numpy as np
import pandas as pd
import os
import cv2
import json
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "/data/aditya/PRISM/")
from utils import run_gd, run_llm, utilities, run_mllm
from dataset.diffimages import run_flux
from dataset.realimages import CC3m_data
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

class pipeline6(object):
    def __init__(self, json_pth):
        f = open(json_pth, "r")
        self.json_obj = json.load(f)

    def forward(self):
        # task 1 - get the nouns and captions
        cn = []
        self.llm_obj = run_llm.run_qwen()
        for i in range(len(self.json_obj)):
            torch.cuda.empty_cache()
            prt = self.json_obj[i]["conversations"][1]['value'].split("\n\n")[0]
            caps, nouns = self.llm_obj.forward(prt.replace("image",""))
            nouns = nouns.split("\n")[1].split(" ")
            cn.append({"nouns":nouns, "caps":caps})
        del self.llm_obj
        torch.cuda.empty_cache()
        with open("/nfshomes/asarkar6/trinity/trinity-data.json", "w") as final:
            json.dump(cn, final)

        # task 2 - get the images
        self.diff_obj = run_flux()
        f = open("/nfshomes/asarkar6/trinity/trinity-data.json", "r")
        json_obj = json.load(f)
        for i in range(len(json_obj)):
            torch.cuda.empty_cache()
            sum_prt = "An image of " + json_obj[i]['caps']
            img_gen = self.diff_obj.forward(sum_prt)
            img_gen.save(os.path.join("/nfshomes/asarkar6/trinity/trinity-images/", str(i)+".png"))
        f.close()
        torch.cuda.empty_cache()
        del self.diff_obj

        # task 3 - get the objects
        self.GD = run_gd.GDINO()
        for i in range(len(json_obj)):
            torch.cuda.empty_cache()
            img_gen = Image.open(os.path.join("/nfshomes/asarkar6/trinity/trinity-images/",str(i)+".png"))
            out = []
            for j in range(json_obj[i]["nouns"]):
                out.append(self.GD.predict(img_gen, json_obj[i]["nouns"][j], 0.3, 0.25,))
            out1 = [{"labels": i["labels"][0], "boxes":i['boxes'].cpu().numpy().tolist()[0]} for i in out if len(i['boxes'].cpu().numpy().tolist()[0]) != 0]
            out_fil = utilities.find_important(out1, img_gen.size)
            json_obj[i]['bbox'] = out_fil
        torch.cuda.empty_cache()

class pipeline7(object):
    def __init__(self):
        self.d = CC3m_data().forward()
        self.qwen_model = run_mllm.run_quen2_vl()
        self.GD = run_gd.GDINO()

    def forward(self):
        for i in list(self.d.keys()):
            img_pth = os.path.join("/fs/nexus-datasets/ConceptualCaptions/training_data_CC3M/images/", i)
            # get dense captions
            messg = get_messages(typ = 0, img_pth = img_pth)
            caps = self.qwen_model.forward(messg)

            # get summarized captions
            messg = get_messages(typ = 1, message = caps[0])
            s_caps = self.qwen_model.forward(messg)

            # get summarized captions
            messg = get_messages(typ = 2, message = s_caps[0])
            nouns = self.qwen_model.forward(messg)
            nouns = nouns[0].split(",")

            # get the bounding boxes
            k = 0
            # [list(cn.keys())]
            for i in nouns:
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
    json_pth = "/nfshomes/asarkar6/trinity/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json"
    pipeline6(json_pth).forward()


