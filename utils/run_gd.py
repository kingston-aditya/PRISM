import torch
from PIL import Image
import cv2
import numpy as np
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
# import sys
# sys.path.insert(1, "/data/aditya/PRISM/")
from utils.utilities import visualize

class GDINO:
    def __init__(self):
        model_id = "IDEA-Research/grounding-dino-base"
        self.device = torch.device("cuda")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def predict(
        self,
        pil_images,
        text_prompt,
        box_threshold,
        text_threshold,
    ):
        for i, prompt in enumerate(text_prompt):
            if prompt[-1] != ".":
                text_prompt[i] += "."
        inputs = self.processor(images=pil_images, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[k.size[::-1] for k in pil_images],
        )
        return results


if __name__ == "__main__":
    gdino_obj = GDINO()
    img = Image.open("/data/aditya/generated_image.png")
    labs = ["cat", "dog"]
    out = gdino_obj.predict([img]*2, labs, 0.3, 0.25,)
    for i in out:
        temp = i['boxes'].cpu().numpy().tolist()[0]
        out_img = visualize(img, {"xmin": int(temp[0]), "ymin": int(temp[1]), "xmax": int(temp[2]), "ymax": int(temp[3])})
        out_img.save("/data/aditya/output_image_"+i["labels"][0]+".png")