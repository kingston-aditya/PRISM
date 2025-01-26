import torch
from PIL import Image
import cv2
import numpy as np
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
import sys
sys.path.insert(1, "/data/aditya/PRISM/")
from utils.utilities import visualize

class GDINO:
    def __init__(self, ckpt_path: str | None = None):
        model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def predict(
        self,
        pil_images: list[Image.Image],
        text_prompt: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict]:
        for i, prompt in enumerate(text_prompt):
            if prompt[-1] != ".":
                text_prompt[i] += "."
        inputs = self.processor(images=pil_images, text=text_prompt, padding = True, return_tensors="pt").to(self.device)
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
    img = Image.open("/data/aditya/visuals1/output_image_0.png")
    print(img.size)
    labs = ["strawberry", "cucumber"]
    out = gdino_obj.predict([img, img], labs, 0.3, 0.25,)
    print(out)
    for i in out:
        if len(i['boxes'].cpu().numpy().tolist()) == 0:
            continue
        else:
            temp = i['boxes'].cpu().numpy().tolist()[0]
            out_img = visualize(img, {"xmin": int(temp[0]), "ymin": int(temp[1]), "xmax": int(temp[2]), "ymax": int(temp[3])})
            out_img.save("/data/aditya/output_image_"+i["labels"][0]+".png")