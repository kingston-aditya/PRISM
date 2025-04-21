def get_config():
    return {
        "repo_path": "/nfshomes/asarkar6/aditya/PRISM/",
        "data_path": "/nfshomes/asarkar6/trinity/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json",
        "llm_model": "Qwen/Qwen2.5-72B-Instruct",
        "cache_dir": "/nfshomes/asarkar6/trinity/model_weights/",
        "batch_size": 128,
        "dataloader_num_workers": 1,
        "is_sdxl": "True",
        "start_len": 0,
        "end_len": 512,
        "output_metadata_folder": "/nfshomes/asarkar6/trinity/finale_data/",
        "output_img_folder": "/nfshomes/asarkar6/trinity/finale_data/images/" 
    }