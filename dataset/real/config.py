# initialize config
def get_config():
    return {
        "repo_path": "/data/home/saividyaranya/PRISM/",
        "input_data_dir": "/fsx/mrs_shlok_sai/cc12m_v2/",
        "llm_model": "Qwen/Qwen2.5-72B-Instruct",
        "mllm_model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "cache_dir": "/data/home/saividyaranya/PRISM/cached_folder_real",
        "batch_size": 512,
        "dataloader_num_workers": 1,
        "is_sdxl": "False",
        "output_metadata_folder": "/data/home/saividyaranya/PRISM/cached_folder_real/metadata_folder",
        "output_img_folder": "/data/home/saividyaranya/PRISM/cached_folder_real/images/",
        "job_id": 1,
        "start_len": 3_000_000,
        "end_len": 4_500_000
    }