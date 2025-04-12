from transformers import AutoProcessor, AutoModelForCausalLM 
import pdb
import torch
from PIL import Image

def parser(caps):
    temp={}
    k=0
    for item in caps:
        t = item.split(".")[0].split(">")[-1]+'.'
        temp[k] = t
        k+=1
    return list(temp.values())

def message_maker(batch):
    temp = []
    for item in batch:
        mssg = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item},
                    {"type": "text", "text": "Caption this image. Caption should be less than 20 words."},
                ],
            }
        ]
        temp.append(mssg)
    return temp

class run_qwen2_vl(object):
    # , args
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.float16, cache_dir = "/nfshomes/asarkar6/trinity/model_weights", device_map="auto")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

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
    
class run_florence(object):
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True, torch_dtype=torch.float16, cache_dir = args.cache_dir).to(self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

    def forward(self, img_input):
        prompt = ["<CAPTION>"]*len(img_input)
        inputs = self.processor(text=prompt, images=img_input, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        parsed_text = parser(generated_text)
        return parsed_text

if __name__ == "__main__":
    img = Image.open("/nfshomes/asarkar6/aditya/test_image.png")
    img1 = Image.open("/nfshomes/asarkar6/trinity/trinity-images/4500.png")
    imgs = [img, img1]*5
    mllm_obj = run_florence()
    caps = mllm_obj.forward(imgs)
    print(len(caps), caps)
    pdb.set_trace()




