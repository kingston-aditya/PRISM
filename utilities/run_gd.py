from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from accelerate import PartialState
from accelerate.utils import gather_object
import pdb

# def correct_inputs(imgs, txts):
#     temp = {}; temp1 = {}
#     k = 0
#     for i in range(len(txts)):
#         for j in txts[i].split(","):
#             temp[k] = imgs[i]
#             temp1[k] = j
#             k+=1
#     return temp, temp1

class GDINO(object):
    def __init__(self, args):
        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir = args["cache_dir"])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir = args["cache_dir"])
        # load it on multiple GPUs
        self.distributed_state = PartialState()
        self.model.to(self.distributed_state.device)

    def predict(
        self,
        pil_images,
        text_prompt,
        box_threshold,
        text_threshold,
    ):
        for i, prompt in enumerate(text_prompt):
            if len(prompt) == 0:
                text_prompt[i] = "."
            else:
                if prompt[-1] != ".":
                    text_prompt[i] += "."

        side = list(zip(pil_images, text_prompt))
        with self.distributed_state.split_between_processes(side) as side_t:
            pi = [item[0] for item in side_t]
            tp = [item[1] for item in side_t]

            batched_inputs = self.processor(images=pi, text=tp, padding = True, return_tensors="pt").to(self.distributed_state.device)
            outputs = self.model(**batched_inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                batched_inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[k.size[::-1] for k in pi],
            )
            for item in results:
                item["scores"] = item["scores"].detach().to("cpu").tolist()
                item["boxes"] = item["boxes"].detach().to("cpu")

        results = gather_object(results)
        return results

# if __name__ == "__main__":
#     args = {"cache_dir": "/nfshomes/asarkar6/trinity/model_weights/"}
#     gdino_obj = GDINO(args)
#     img = Image.open("/nfshomes/asarkar6/aditya/test_image.png")
#     img1 = Image.open("/nfshomes/asarkar6/trinity/trinity-images/4500.png")

#     fin_out = {}; k=0

#     # total size 4
#     imgs = [img, img1]*2
#     labs = ["lion", "road"]*2
#     out = gdino_obj.predict(imgs, labs, 0.3, 0.25,)
    
    