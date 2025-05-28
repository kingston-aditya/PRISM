# initialize config
def get_config():
    return {
        "repo_path": "/nfshomes/asarkar6/aditya/PRISM/",
        "dataset_name": "/nfshomes/asarkar6/trinity/train_data/",
        "copy_num": 30,
        "llm_model": "Qwen/Qwen2.5-7B-Instruct",
        "cache_dir": "/nfshomes/asarkar6/trinity/model_weights/",
        "batch_size": 4,
        "dataloader_num_workers": 1,
    }