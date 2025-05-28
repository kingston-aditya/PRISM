# def get_config():
#     return {
#         "repo_path": "/data/home/saividyaranya/PRISM/",
#         "data_path": "/data/home/saividyaranya/PRISM/share-captioner_coco_lcs_sam_1246k_1107.json",
#         "llm_model": "Qwen/Qwen2.5-72B-Instruct",
#         "cache_dir": "/data/home/saividyaranya/PRISM/cached_folder_syn2",
#         "batch_size": 128,
#         "dataloader_num_workers": 1,
#         "is_sdxl": "False",
#         "start_len": 0,
#         "end_len": 25_000,
#         "output_metadata_folder": "/data/home/saividyaranya/PRISM/cached_folder_syn2/metadata_folder",
#         "output_img_folder": "/data/home/saividyaranya/PRISM/cached_folder_syn2/images/" ,
#         "job_id": 1,
#         "copy_num": 30,
#     }

def get_config():
    return {
        "repo_path": "/nfshomes/asarkar6/aditya/PRISM/",
        "data_path": "/nfshomes/asarkar6/trinity/train_data/",
        "copy_num": 30,
        "llm_model": "Qwen/Qwen2.5-7B-Instruct",
        "cache_dir": "/nfshomes/asarkar6/trinity/model_weights/",
        "batch_size": 4,
    }


# def get_config_2():
#     return {
#         "repo_path": "/data/home/saividyaranya/PRISM/",
#         "data_path": "/data/home/saividyaranya/PRISM/share-captioner_coco_lcs_sam_1246k_1107.json",
#         "llm_model": "Qwen/Qwen2.5-72B-Instruct",
#         "cache_dir": "/data/home/saividyaranya/PRISM/cached_folder2",
#         "batch_size": 512,
#         "dataloader_num_workers": 1,
#         "is_sdxl": "False",
#         "start_len": 25_000,
#         "end_len": 50_000,
#         "output_metadata_folder": "/data/home/saividyaranya/PRISM/cached_folder2/metadata_folder",
#         "output_img_folder": "/data/home/saividyaranya/PRISM/cached_folder2/images/" ,
#         "job_id": 2
#     }

# def get_config_3():
#     return {
#         "repo_path": "/data/home/saividyaranya/PRISM/",
#         "data_path": "/data/home/saividyaranya/PRISM/share-captioner_coco_lcs_sam_1246k_1107.json",
#         "llm_model": "Qwen/Qwen2.5-72B-Instruct",
#         "cache_dir": "/data/home/saividyaranya/PRISM/cached_folder2",
#         "batch_size": 512,
#         "dataloader_num_workers": 1,
#         "is_sdxl": "False",
#         "start_len": 50_000,
#         "end_len": 75_000,
#         "output_metadata_folder": "/data/home/saividyaranya/PRISM/cached_folder2/metadata_folder",
#         "output_img_folder": "/data/home/saividyaranya/PRISM/cached_folder2/images/" ,
#         "job_id": 3
#     }

# def get_config_4():
#     return {
#         "repo_path": "/data/home/saividyaranya/PRISM/",
#         "data_path": "/data/home/saividyaranya/PRISM/share-captioner_coco_lcs_sam_1246k_1107.json",
#         "llm_model": "Qwen/Qwen2.5-72B-Instruct",
#         "cache_dir": "/data/home/saividyaranya/PRISM/cached_folder2",
#         "batch_size": 512,
#         "dataloader_num_workers": 1,
#         "is_sdxl": "False",
#         "start_len": 75_000,
#         "end_len": 100_000,
#         "output_metadata_folder": "/data/home/saividyaranya/PRISM/cached_folder2/metadata_folder",
#         "output_img_folder": "/data/home/saividyaranya/PRISM/cached_folder2/images/" ,
#         "job_id": 4
#     }


# def get_config_5():
#     return {
#         "repo_path": "/data/home/saividyaranya/PRISM/",
#         "data_path": "/data/home/saividyaranya/PRISM/share-captioner_coco_lcs_sam_1246k_1107.json",
#         "llm_model": "Qwen/Qwen2.5-72B-Instruct",
#         "cache_dir": "/data/home/saividyaranya/PRISM/cached_folder2",
#         "batch_size": 512,
#         "dataloader_num_workers": 1,
#         "is_sdxl": "False",
#         "start_len": 100_000,
#         "end_len": 125_000,
#         "output_metadata_folder": "/data/home/saividyaranya/PRISM/cached_folder2/metadata_folder",
#         "output_img_folder": "/data/home/saividyaranya/PRISM/cached_folder2/images/" ,
#         "job_id": 5
#     }