import os
import numpy as np
from PIL import Image
import pandas as pd

class real_data(object):
    def __init__(self):
        self.fil_pth = ""
        self.cap_pth = None
    
    def forward(self):
        d = {}
        if self.cap_pth is not None:
            df = pd.read_csv(self.cap_pth, dtype=object)
            sze = 20000
            for i in range(3*sze,5*sze):
                try:
                    img = Image.open(os.path.join(self.fil_pth, df.iat[i,0]))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    d[df.iat[i,0]] = img
                except Exception as e:
                    pass
        return d

class CC3m_data(real_data):
    def __init__(self):
        super().__init__()
        self.fil_pth = "/fs/nexus-datasets/ConceptualCaptions/training_data_CC3M/images/"
        self.cap_pth = "/fs/nexus-datasets/ConceptualCaptions/training_data_CC3M/train_filtered.csv"

if __name__ == "__main__":
    c = CC3m_data().forward()
