export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"

accelerate launch --mixed_precision="fp16" --multi_gpu --main_process_port 29501 /data/home/saividyaranya/PRISM/models/infer_trinity_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
   --dataset_name="/data/home/saividyaranya/PRISM/cached_folder_real/images_again/" \
  --output_dir="/data/home/saividyaranya/PRISM/all_output_logs/sdxl15" \
 --backup="/home/saividyaranya/PRISM/backup/" \
    --valid_path_name="/data/home/saividyaranya/PRISM/validation/" \
 --wanna_bg=1 \
 --resume_from_checkpoint="latest" \
 --resolution=512 \
  --cache_dir="/data/home/saividyaranya/PRISM/model_weights/" \
 --learning_rate=1e-5 \
 --train_batch_size=2 \
 --mixed_precision="fp16" \
 --num_validation_images=4 