import argparse
from PRISM.dataset.data_create.data_gen import pipeline6, pipeline7

def run_model(args):
    if args.typ == 6:
        json_pth = "/nfshomes/asarkar6/trinity/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json"
        pipeline6(json_pth).forward()
    elif args.typ == 7:
        pipeline7().forward()
    else:
        print("idk")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'gets the json file given a context')
    parser.add_argument("typ", type=int, help='what pipeline to run')
    args = parser.parse_args()
    run_model(args)

