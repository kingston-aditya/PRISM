import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import os
from PIL import Image

import pdb

ANNO_FILE = '/fs/cml-datasets/coco/annotations/instances_val2017.json'
CAPTION_FILE = '/fs/cml-datasets/coco/annotations/captions_val2017.json'
IMAGE_DIR = '/fs/cml-datasets/coco/images/val2017'

def get_annotations(coco, coco_cap, image_id):
    img_info = coco.loadImgs(image_id)[0]
    image_path = os.path.join(IMAGE_DIR, img_info['file_name'])

    # load the image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)

    # load the annotations
    ann_ids = coco.getAnnIds(imgIds=image_id, iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    # load captions
    ann_cap_ids = coco_cap.getAnnIds(imgIds=image_id)
    caps = coco_cap.loadAnns(ann_cap_ids)[0]['caption']

    # get the segmentation mask
    height, width = image.shape[0], image.shape[1]
    segmentation_mask = np.zeros((height, width), dtype=np.uint8)

    def get_bounding_box(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return y_min, y_max, x_min, x_max

    temp = []; max_area = -9999
    for ann in anns:
        segmentation_mask = coco.annToMask(ann)
        y_min, y_max, x_min, x_max = get_bounding_box(segmentation_mask)
        temp.append({"ymin": y_min, "ymax": y_max, "xmin": x_min, "xmax":x_max, "area": (x_max-x_min)*(y_max-y_min), "caption": caps})
        if (x_max-x_min)*(y_max-y_min)/(height*width) > max_area:
            max_area = (x_max-x_min)*(y_max-y_min)/(height*width)
    
    return temp, max_area, Image.fromarray(image)

# concatenate a list of PIL images
def concatenate_images(objects, bg=None, direction="horizontal"):
    if not objects:
        return None
    
    # Filter out None images
    valid_images = [Image.fromarray(np.asarray(obj_item["image"])[int(obj_item["ymin"]): int(obj_item["ymax"]), int(obj_item["xmin"]): int(obj_item["xmax"])]) for obj_item in objects]
    
    if not valid_images:
        return None
    
    if len(valid_images) == 1:
        return valid_images[0].convert("RGB")
    
    # Convert all images to RGB
    valid_images = [img.convert("RGB") for img in valid_images]
    
    if direction == "horizontal":
        # Calculate total width and max height
        total_width = sum(img.width for img in valid_images)
        max_height = max(img.height for img in valid_images)
        
        # Create new image
        if bg is None:
            concatenated = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        else:
            concatenated = bg.resize((total_width, max_height))
        
        # Paste images
        x_offset = 0
        for img in valid_images:
            # Center image vertically if heights differ
            y_offset = (max_height - img.height) // 2
            concatenated.paste(img, (x_offset, y_offset))
            x_offset += img.width
            
    else:  
        # Calculate max width and total height
        max_width = max(img.width for img in valid_images)
        total_height = sum(img.height for img in valid_images)
        
        # Create new image
        concatenated = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        
        # Paste images
        y_offset = 0
        for img in valid_images:
            # Center image horizontally if widths differ
            x_offset = (max_width - img.width) // 2
            concatenated.paste(img, (x_offset, y_offset))
            y_offset += img.height
    
    return concatenated

def get_coco_objects():
    coco = COCO(ANNO_FILE)

    coco_cap = COCO(CAPTION_FILE)

    img_ids = coco.getImgIds()
    image_idxs = np.random.choice(len(img_ids), size=100, replace=False)

    # get the objects
    objects = []
    for _, image_idx in enumerate(image_idxs):
        try: 
            # get the image path
            image_id = img_ids[image_idx]
            temp, max_area, image = get_annotations(coco, coco_cap, image_id)
            if max_area < 0.45:
                raise ValueError("Objects are too small")
        except: 
            while max_area > 0.45:
                new_image_idx = np.random.randint(0, len(img_ids))
                image_id = img_ids[new_image_idx]
                temp, max_area, image = get_annotations(coco, coco_cap, image_id)
                
        temp = sorted(temp, key=lambda x: x['area'], reverse=True)
        objects.append([temp[i].update({"image": image}) for i in range(2)])

        return objects

if __name__ == "__main__":
    # get the 100 objects
    objects = get_coco_objects()

    for idx, item in enumerate(objects):
        # 1. get the baseline images

        # create baseline 1 - ground truth image
        gt_image = item["image"]

        # create baseline 2 - concatenated image
        concat_image_white = concatenate_images(objects)

        # create baseline 3_1 - concatenated image with background 1
        bg1 = Image.open("5.png")
        concat_image_bg1 = concatenate_images(objects, bg1)

        # create baseline 3_2 - concatenated image with background 2
        bg2 = Image.open("foster-lake.jpg")
        concat_image_bg2 = concatenate_images(objects, bg2)

        # 2. get the baseline images



        








