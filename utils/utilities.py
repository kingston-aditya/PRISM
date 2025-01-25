import numpy as np
import cv2
from PIL import Image

def visualize(img, annot):
    img_cv2 = np.array(img)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
    for det in annot:
        color = np.array([0, 0, 255])
        cv2.rectangle(img_cv2, (annot['xmin'], annot['ymin']), (annot['xmax'], annot['ymax']), color.tolist(), 2)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_cv2)