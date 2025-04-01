from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import pdb
import torch

def message_maker(batch):
    temp = []
    for item in batch:
        mssg = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item},
                    {"type": "text", "text": "Caption this image in detail. Caption should be less than 40 words."},
                ],
            }
        ]
        temp.append(mssg)
    return temp

class run_qwen2_vl(object):
    # , args
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.mllm_model, torch_dtype="auto", cache_dir = args.cache_dir, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(args.mllm_model)

    def forward(self, img_pths):
        messages = message_maker(img_pths)
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side='left',
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Batch Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_texts[0] = output_texts[0].split("\n")[-1]
        return output_texts
        

if __name__ == "__main__":
    mllm_obj = run_qwen2_vl()
    img = "/nfshomes/asarkar6/aditya/generated_image.png"
    img1 = "/nfshomes/asarkar6/trinity/trinity-images/4500.png"
    imgs = [img, img1]
    caps = mllm_obj.forward(imgs)
    print(len(caps), caps)
    pdb.set_trace()




