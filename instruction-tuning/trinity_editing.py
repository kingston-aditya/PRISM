import json
from tqdm import tqdm
import os

inp_dir = ""

if __name__ == "__main__":
    # TODO - Data loader
    edit_dataloader = []

    with open(os.path.join(inp_dir,"metaedit.jsonl"), "w") as f:
        # save the dataset
        for idx, item in tqdm(enumerate(edit_dataloader), desc="Saving data"):
            # load the dataset batches
            inp_imgs_pth = item["inp_imgs"]
            out_imgs_pth = item["out_imgs"]
            prts = item["prompt"]

            # create data
            temp = {"file_name": out_imgs_pth, "prompt": prts, "object": [{"img_pth": inp_imgs_pth}]}
            f.write(json.dumps(temp) + '\n')

        