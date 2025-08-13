export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"

accelerate launch --mixed_precision="fp16" --multi_gpu --main_process_port 29501 /nfshomes/asarkar6/aditya/PRISM/models/infer_pix2pix_trinity.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name="/nfshomes/asarkar6/trinity/train_data/" \
    --output_dir="/nfshomes/asarkar6/scratch/test_image/" \
    --cache_dir="/nfshomes/asarkar6/trinity/model_weights/" \
    --backup="/nfshomes/asarkar6/aditya/PRISM/backup/" \
    --output_img_dir="/nfshomes/asarkar6/aditya/gen_images/" \
    --valid_path_name="/nfshomes/asarkar6/aditya/PRISM/validation/" \
    --num_validation_images=1 \
    --resume_from_checkpoint="latest" \
    --resolution=256 \
    --random_flip \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=40 \
    --learning_rate=5e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=100 \
    --lr_scheduler="cosine" \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision="fp16" \
    --seed=42