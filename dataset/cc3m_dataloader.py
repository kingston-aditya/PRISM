from datasets import load_dataset, Image
from glob import glob
import torch
import PIL
import os
import io
# import pdb

def t2i_process_fn(batch):
    images = batch["image"]
    captions = batch["caption"]
    for i in range(len(images)):
        try:
            images[i] = PIL.Image.open(io.BytesIO(images[i]["bytes"]) if images[i]["bytes"] is not None else images[i]["path"]).convert('RGB')
        except:
            print("corrupt!!!!")
            images[i] = None
            captions[i] = ""

    while all(x is None for x in batch["image"]):
        randidx = torch.randint(0, len(train_dataset), (1,)).item()
        batch = train_dataset[randidx]
        # Expand single items into batches
        batch = {key: [value] for key, value in batch.items()}

    return batch
    # return batch["image"], batch["caption"]

def return_cc3_train_dataset(args):
    data_files = glob(os.path.join(args.input_data_dir, "*.tar"))
    train_dataset = load_dataset(
        "webdataset",
        data_files=data_files,
        cache_dir=args.cache_dir,
        split="train",
    )

    train_dataset = train_dataset.rename_column("jpg", "image")
    train_dataset = train_dataset.rename_column("txt", "caption")
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (["image", "caption"])])


    train_dataset = train_dataset.cast_column("image", Image(decode=False))
    train_dataset.set_transform(t2i_process_fn)

    return train_dataset