export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="/nfshomes/asarkar6/scratch/test_image/"

accelerate launch --mixed_precision="fp16" --multi_gpu --main_process_port 29501 /nfshomes/asarkar6/aditya/PRISM/models/infer_trinity_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="/nfshomes/asarkar6/trinity/train_data/" \
 --valid_path_name="/nfshomes/asarkar6/aditya/PRISM/validation/" \
 --backup="/nfshomes/asarkar6/aditya/PRISM/backup/" \
 --output_img_dir="/nfshomes/asarkar6/aditya/gen_images/" \
 --wanna_bg=1 \
 --resume_from_checkpoint="latest" \
 --resolution=512 \
 --cache_dir="/nfshomes/asarkar6/trinity/model_weights/" \
 --learning_rate=1e-5 \
 --train_batch_size=2 \
 --mixed_precision="fp16" \
 --num_validation_images=4 