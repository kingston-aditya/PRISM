import numpy as np
import cv2
from PIL import Image
import pandas as pd

def visualize(img, annot):
    img_cv2 = np.array(img)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
    for det in annot:
        color = np.array([0, 0, 255])
        cv2.rectangle(img_cv2, (annot['xmin'], annot['ymin']), (annot['xmax'], annot['ymax']), color.tolist(), 2)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_cv2)

def find_area(df, img_shp):
    segs = []
    for i in range(df.shape[0]):
        mask = np.ones((img_shp[0], img_shp[1]))
        mask[int(df.iloc[i]['YMin']*img_shp[0]):int(df.iloc[i]['YMax']*img_shp[0]), int(df.iloc[i]['XMin']*img_shp[1]):int(df.iloc[i]['XMax']*img_shp[1])] = 0
        segs.append(1.0-mask)
    return np.sum(1.0-mask)/(img_shp[0]*img_shp[1]), segs

class compute_area(object):
    def __init__(self, img_rgb_shp, img_id, csv_pth):
        df = pd.read_csv(csv_pth)
        self.main_df = df[df['ImageID'] == img_id]
        temp = []
        for i in range(self.main_df.shape[0]):
            temp.append(int((self.main_df.iloc[i]['XMax'] - self.main_df.iloc[i]['XMin'])*(self.main_df.iloc[i]['YMax'] - self.main_df.iloc[i]['YMin'])*img_rgb_shp[0]*img_rgb_shp[1]))
        self.main_df["area"] = temp
        self.main_df = self.main_df.sort_values(by="area", ascending=False)
        self.img_shp = img_rgb_shp
    
    def check(self):
        if self.main_df.shape[0] < 2:
            return None
        else:
            self.k = min(self.main_df.shape[0], 3)
        portion, segs = find_area(self.main_df.iloc[:self.k], self.img_shp)
        self.segs = segs
        return portion
        
    def get_segs(self, type_df, img_rgb):
        nouns = []
        seg_img = []
        for i in range(self.k):
            nouns.append(type_df[type_df['LabelName']==self.main_df.iloc[i]['LabelName']].iloc[0]['DisplayName'])
            seg_img.append(img_rgb[int(self.main_df.iloc[i]['YMin']*self.img_shp[0]):int(self.main_df.iloc[i]['YMax']*self.img_shp[0]), int(self.main_df.iloc[i]['XMin']*self.img_shp[1]):int(self.main_df.iloc[i]['XMax']*self.img_shp[1]),:])
        return seg_img, nouns
