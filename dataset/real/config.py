def get_config():
    return {
        "repo_path": "/data/aditya/PRISM/",
        "input_data_dir": "/mnt/ssd/",
        "llm_model": "Qwen/Qwen2.5-7B-Instruct",
        "mllm_model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "cache_dir": "/nfshomes/asarkar6/trinity/model_weights/",
        "batch_size": 4,
        "dataloader_num_workers": 1,
        "is_sdxl": "True",
        "start_len": 0,
        "end_len": 16,
        "output_metadata_folder": "/mnt/ssd/finale_data/",
        "output_img_folder": "/mnt/ssd/finale_data/images/" 
    }