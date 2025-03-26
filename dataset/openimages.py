import cv2
import pandas as pd
import sys
sys.path.insert(1, "/data/aditya/PRISM/utils/")
from compute_area import compute_area

class openimages(object):
    def __init__(self, img_pth, csv_pth):
        self.img_pth = img_pth
        self.img_id = img_pth.split(".")[0].split("/")[-1]
        self.csv_pth = csv_pth

    def read_data(self):
        img_bgr = cv2.imread(self.img_pth)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb, img_rgb.shape
    
    def forward(self):
        img_rgb, img_rgb_shp = self.read_data()
        noun_match = pd.read_csv("/data/datasets/openimages/oidv7-class-descriptions-boxable.csv", dtype=object)
        area_comp = compute_area(img_rgb_shp, self.img_id, self.csv_pth)
        if area_comp.check() == None:
            return None
        elif area_comp.check() > 0.3:
            segs, nouns = area_comp.get_segs(noun_match, img_rgb)
            return img_rgb, segs, nouns
        else:
            return None




