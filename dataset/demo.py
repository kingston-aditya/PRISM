from PIL import Image
import numpy as np
import json
import os
import glob

def read_eval_dataset():
    # read multiple files
    json_obj = {"image":[], "prompt":[], "object":[]}

    for name in glob.glob("/nfshomes/asarkar6/aditya/PRISM/validation/*.jsonl"):
        with open(os.path.join("/nfshomes/asarkar6/aditya/PRISM/validation/", name), "r") as f:
            for line in f:
                try:
                    temp = json.loads(line.strip())
                    # saves the image
                    json_obj["image"].append(temp["file_name"])
                    # saves the text prompt
                    json_obj["prompt"].append(temp["prompt"])
                    # saves the object
                    if temp["object"] is not None:
                        json_obj["object"].append(temp["object"])
                    else:
                        json_obj["object"].append([])
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON for line: {line.strip()} with error: {e}")
    return json_obj

def main(output_dir):
    json_obj = read_eval_dataset()
    for idx, img_pth in enumerate(json_obj["image"]):
        # create a new directory
        save_pth = os.path.join(output_dir, str(idx)+"_sample")
        os.makedirs(save_pth, exist_ok=True)

        # read the main image
        main_img = Image.open(img_pth).convert("RGB")

        # create the bounding boxes
        if len(json_obj["object"][idx])!=0:
            for k in json_obj["object"][idx]:
                obj_img = Image.fromarray(np.asarray(main_img)[k["ymin"]:k["ymax"], k["xmin"]:k["xmax"]])
                obj_img.save(os.path.join(save_pth, str(k["labels"])+".jpg"))
        else:
            pass
    
if __name__ == "__main__":
    main("/nfshomes/asarkar6/aditya/sample_images/")
