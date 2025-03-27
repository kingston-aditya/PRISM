python /nfshomes/asarkar6/aditya/PRISM/dataset/datagen_syn.py \
  --llm_model="Qwen/Qwen2.5-7B-Instruct" \
  --cache_dir="/nfshomes/asarkar6/trinity/model_weights/" \
  --batch_size=2 \
  --total_length_yes="False" \
  --total_length_no=10 \
  --dataloader_num_workers=2 \
  --is_sdxl="True" \
  --output_img_folder="/nfshomes/asarkar6/trinity/finale_data/images" \
  --output_metadata_folder="/nfshomes/asarkar6/trinity/finale_data/"
