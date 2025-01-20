import torch
import numpy as np
import pandas as pd
import os
import cv2
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, "/data/aditya/PRISM/")
from dataset.openimages import openimages
from utils.retrieval import retrieve_img

def main(dir_pth, csv_pth):
    k=0
    for i in os.listdir(dir_pth):
        img_pth = os.path.join(dir_pth, i)
        obj = openimages(img_pth, csv_pth)
        if obj.forward()==None:
            k+=1
            print("Done",k)
            continue
        else:
            img_rgb, segs, nouns = obj.forward()
            new_dir_pth = os.path.join("/data/aditya/visuals/",str(k+1))
            os.makedirs(new_dir_pth, exist_ok = True)
            cv2.imwrite(os.path.join(new_dir_pth,'final_img.jpg'), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            for j in range(len(segs)):
                cv2.imwrite(os.path.join(new_dir_pth,'seg_img_'+str(nouns[j])+"_"+str(j)+'.jpg'), cv2.cvtColor(segs[j], cv2.COLOR_RGB2BGR))
                obj1 = retrieve_img(segs[j]).retrieve_X()
                print(obj1)
                j+=1
            k+=1
            print("Done",k)

if __name__ == "__main__":
    dir_pth = "/data/datasets/openimages/images/" 
    csv_pth = "/data/datasets/openimages/small_op_annotations.csv"
    main(dir_pth, csv_pth)
