# initialize config
def get_config():
    return {
        "repo_path": "/nfshomes/asarkar6/aditya/PRISM/",
        "input_data_dir": "/fsx/mrs_shlok_sai/cc12m_v2/",
        "llm_model": "Qwen/Qwen2.5-72B-Instruct",
        "mllm_model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "cache_dir": "/nfshomes/asarkar6/trinity/model_weights/",
        "batch_size": 512,
        "dataloader_num_workers": 1,
        "is_sdxl": "False",
        "output_metadata_folder": "/nfshomes/asarkar6/trinity/finale_data/",
        "output_img_folder": "/nfshomes/asarkar6/trinity/finale_data/images/",
        "job_id": "",
        "start_len": 3_000_000,
        "end_len": 4_500_000
    }