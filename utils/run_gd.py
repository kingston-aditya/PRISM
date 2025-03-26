import torch
from PIL import Image
import cv2
import numpy as np
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
import pdb

def correct_inputs(imgs, txts):
    temp = {}
    for i in range(len(txts)):
        for j in txts[i].split(","):
            temp[j] = imgs[i]
    return temp

class GDINO(object):
    def __init__(self, args, ckpt_path: str | None = None):
        model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda"
        # cache_dir = args.cache_dir
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir = args.cache_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir = args.cache_dir).to(self.device)

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
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[k.size[::-1] for k in pil_images],
        )
        return results

# if __name__ == "__main__":
#     gdino_obj = GDINO()
#     img = Image.open("/nfshomes/asarkar6/aditya/generated_image.png")
#     img1 = Image.open("/nfshomes/asarkar6/trinity/trinity-images/4500.png")

#     imgs = [img, img1]
#     labs = ["strawberry", "human"]
#     temp = correct_inputs(imgs, labs)

#     out = gdino_obj.predict(list(temp.values()), list(temp.keys()), 0.3, 0.25,)
#     pdb.set_trace()