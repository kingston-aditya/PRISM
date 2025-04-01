python /nfshomes/asarkar6/aditya/PRISM/dataset/datagen_real.py \
  --llm_model="Qwen/Qwen2.5-7B-Instruct" \
  --mllm_model="Qwen/Qwen2.5-VL-7B-Instruct" \
  --cache_dir="/nfshomes/asarkar6/trinity/model_weights/" \
  --total_length=2000000 \
  --batch_size=2 \
  --dataloader_num_workers=2 \
  --output_img_folder="/nfshomes/asarkar6/trinity/finale_data/images" \
  --output_metadata_folder="/nfshomes/asarkar6/trinity/finale_data/"