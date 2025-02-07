from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

class make_prt_qwen(object):
    def make_captions(self, img_pth):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img_pth,}, 
            {"type": "text", "text": "Dense caption this image. The caption should contain all objects present in the image."},],}]
        return messages
    
    def make_summary(self, prt):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prt,}, 
            {"type": "text", "text": "Summarize the given text to a maximum of 30 words. The output texts should contain all objects."},],}]
        return messages
    
    def find_nouns(self, prt):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "You are an AI assistant. An image is generated using the given caption. Output objects that can be present in the generated image and there is no overlap between the bounding boxes on those objects. Objects should be from the given text and must be described in less than 4 words. Abstract nouns should be given as output.",}, 
            {"type": "text", "text": "Here are two examples.",},
            {"type": "text", "text": "Text - An dog is playing with a cat in a lush green mountainous background. \n\n Answer - dog, cat \n\n Explanation - It is easy to make bounding boxes on dog and cat, but it's not possible to make it on a background.",},
            {"type": "text", "text": "Text - Horses and Zebras are standing in a green grass and expressing sadness with activity. \n\n Answer - horses, zebras \n\n Explanation - It is easy to make bounding boxes on horses and zebra, but it's not possible to make it on a background. Also Sadness and activity are abstract nouns, hence not considered.",},
            {"type": "text", "text": "Do NOT output explanatin. Text - "+prt+"\n\n Answer -",},],}]
        return messages

def get_messages(typ, message=None, img_pth=None):
        if typ == 0:
            messages = make_prt_qwen().make_captions(img_pth)
        elif typ == 1:
            messages = make_prt_qwen().make_summary(message)
        elif typ == 2:
            messages = make_prt_qwen().find_nouns(message)
        else:
            print("Dont know!")
        return messages

class run_quen2_vl(object):
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    def forward(self, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

if __name__ == "__main__":
    img_pth = "/fs/nexus-datasets/ConceptualCaptions/training_data_CC3M/images/images/2901536091.png"
    qwen_model = run_quen2_vl()
    # get dense captions
    messg = get_messages(typ = 0, img_pth = img_pth)
    caps = qwen_model.forward(messg)
    print("CAPTION", caps)

    # get summarized captions
    messg = get_messages(typ = 1, message = caps[0])
    s_caps = qwen_model.forward(messg)
    print("S. CAPTION", s_caps)

    # get summarized captions
    messg = get_messages(typ = 2, message = s_caps[0])
    nouns = qwen_model.forward(messg)
    print("NOUNS", nouns[0].split(","))



