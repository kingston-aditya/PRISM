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

def calculate_iou(box1, box2, img_shape):
    mask1 = np.zeros(img_shape, dtype=np.uint8)
    mask2 = np.zeros(img_shape, dtype=np.uint8)
    mask1[box1["ymin"]:box1["ymax"], box1["xmin"]:box1["xmax"]] = 1
    mask2[box2["ymin"]:box2["ymax"], box2["xmin"]:box2["xmax"]] = 1
    intersect = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))
    if union == 0:
        return 0
    else:
        return intersect/union

def find_important(out, img_shape):
    min_overlap = float('inf')
    best_triplet = None

    n = len(out)

    # Iterate over all possible triplets
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                box1 = {"xmin":int(out[i]["boxes"][0]),"xmax":int(out[i]["boxes"][2]), "ymin":int(out[i]["boxes"][1]), "ymax": int(out[i]["boxes"][3]), "labels":out[i]["labels"]}
                box2 = {"xmin":int(out[j]["boxes"][0]),"xmax":int(out[j]["boxes"][2]), "ymin":int(out[j]["boxes"][1]), "ymax": int(out[j]["boxes"][3]), "labels":out[j]["labels"]}
                box3 = {"xmin":int(out[k]["boxes"][0]),"xmax":int(out[k]["boxes"][2]), "ymin":int(out[k]["boxes"][1]), "ymax": int(out[k]["boxes"][3]), "labels":out[k]["labels"]}

                # Calculate pairwise IoUs
                iou12 = calculate_iou(box1, box2, img_shape)
                iou13 = calculate_iou(box1, box3, img_shape)
                iou23 = calculate_iou(box2, box3, img_shape)

                # Calculate the sum of IoUs for the triplet
                overlap_sum = (iou12 + iou13 + iou23)/3

                # Check if this triplet has the minimum overlap
                if overlap_sum < min_overlap:
                    min_overlap = overlap_sum
                    best_triplet = [box1, box2, box3]

    return best_triplet