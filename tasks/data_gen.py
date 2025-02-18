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
sys.path.insert(0, "/nfshomes/asarkar6/PRISM/")
from utils import run_gd, run_llm, utilities, run_mllm
from dataset.diffimages import run_sdxl, run_flux
from dataset.realimages import CC3m_data

class pipeline6(object):
    def __init__(self, json_pth):
        f = open(json_pth, "r")
        self.json_obj = json.load(f)
        f.close()

    def forward(self):
        f1 = open("/nfshomes/asarkar6/PRISM/output.txt","w")
        # task 1 - get the nouns and captions
        # cn = []
        # self.llm_obj = run_llm.run_qwen()
        # for i in range(20000):
        #     f1.write("TS"+"\t"+str(i))
        #     torch.cuda.empty_cache()
        #     prt = self.json_obj[i]["conversations"][1]['value'].split("\n\n")[0]
        #     caps, nouns = self.llm_obj.forward(prt.replace("image",""))
        #     nouns = nouns.split(" ")
        #     cn.append({"nouns":nouns, "caps":caps})
        #     with open("/nfshomes/asarkar6/trinity/trinity-data.json", "w") as final:
        #         json.dump(cn, final)
        # del self.llm_obj
        # torch.cuda.empty_cache()
        
        # task 2 - get the images
        self.diff_obj = run_sdxl()
        f = open("/nfshomes/asarkar6/trinity/trinity-data.json", "r")
        json_obj = json.load(f)
        for i in range(645,len(json_obj)):
            f1.write("IG"+"\t"+str(i))
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
            f1.write("BB"+"\t"+str(i))
            torch.cuda.empty_cache()
            img_gen = Image.open(os.path.join("/nfshomes/asarkar6/trinity/trinity-images/",str(i)+".png"))
            out = []
            for j in range(len(json_obj[i]["nouns"])):
                out.append(self.GD.predict([img_gen], [json_obj[i]["nouns"][j]], 0.3, 0.25,))
            out1 = []
            for k in out:
                if len(k[0]['boxes'].cpu().numpy().tolist()) != 0:
                    out1.append({"labels": k[0]['text_labels'][0], "boxes": k[0]['boxes'].cpu().numpy().tolist()[0]})
            out_fil = utilities.find_important(out1, img_gen.size)
            json_obj[i]['bbox'] = out_fil
            with open("/nfshomes/asarkar6/trinity/trinity-data.json", "w") as final:
                json.dump(json_obj, final)
        torch.cuda.empty_cache()
        f1.close()

class pipeline7(object):
    def __init__(self):
        self.d = CC3m_data().forward()

    def forward(self):
        cn = []
        k=0
        f1 = open("/nfshomes/asarkar6/PRISM/output_1.txt","w")

        # Task 1 - Get the summarized captions.
        self.qwen_model = run_mllm.run_quen2_vl()
        for i in list(self.d.keys()):
            f1.write("TS"+"\t"+str(k))
            img_pth = os.path.join("/fs/nexus-datasets/ConceptualCaptions/training_data_CC3M/images/", i)
            torch.cuda.empty_cache()

            # get captions
            messg = run_mllm.get_messages(typ = 0, img_pth = img_pth)
            caps = self.qwen_model.forward(messg)

            # get nouns
            messg = run_mllm.get_messages(typ = 1, message = caps[0])
            nouns = self.qwen_model.forward(messg)
            nouns = nouns[0].split(",")

            cn.append({"caps": caps, "nouns": nouns, "img_pth":img_pth})
            k+=1

            with open("/nfshomes/asarkar6/trinity/trinity-data-real3.json", "w") as final:
                json.dump(cn, final)
        torch.cuda.empty_cache()
        del self.qwen_model

        # Task 2 - Get the bounding boxes.
        f = open("/nfshomes/asarkar6/trinity/trinity-data-real3.json", "r")
        json_obj = json.load(f)
        f.close()

        self.GD = run_gd.GDINO()
        # len(json_obj)
        for i in range(len(json_obj)):
            f1.write("BB"+"\t"+str(i))
            print("BB"+"\t"+str(i))
            torch.cuda.empty_cache()
            img_gen = Image.open(json_obj[i]["img_pth"])
            out = []
            if len(json_obj[i]["nouns"]) >= 3:
                for j in range(len(json_obj[i]["nouns"])):
                    out.append(self.GD.predict([img_gen], [json_obj[i]["nouns"][j]], 0.3, 0.25,))
                out1 = []
                for k in out:
                    if len(k[0]['boxes'].cpu().numpy().tolist()) != 0:
                        out1.append({"labels": k[0]['text_labels'][0], "boxes": k[0]['boxes'].cpu().numpy().tolist()[0]})
                out_fil = utilities.find_important(out1, img_gen.size)
                json_obj[i]['bbox'] = out_fil
                with open("/nfshomes/asarkar6/trinity/trinity-data-real3.json", "w") as final:
                    json.dump(json_obj, final)
            else:
                for j in range(len(json_obj[i]["nouns"])):
                    out.append(self.GD.predict([img_gen], [json_obj[i]["nouns"][j]], 0.3, 0.25,))
                out1 = []
                for k in out:
                    if len(k[0]['boxes'].cpu().numpy().tolist()) != 0:
                        out1.append({"labels": k[0]['text_labels'][0], "boxes": k[0]['boxes'].cpu().numpy().tolist()[0]})
                json_obj[i]['bbox'] = out1
                with open("/nfshomes/asarkar6/trinity/trinity-data-real3.json", "w") as final:
                    json.dump(json_obj, final)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # for pipeline 5
    json_pth = "/nfshomes/asarkar6/trinity/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json"
    pipeline7().forward()


