export DATASET_CONFIG="/nfshomes/asarkar6/aditya/PRISM/dreamengine/src/configs/ti2i_jdb1M.json"
export OUTPUT_DIR="/nfshomes/asarkar6/trinity/model_weights/"

export TOKENIZERS_PARALLELISM=false

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# --main_process_port 29501

# run this part for stage 1 training.
# accelerate launch --num_processes 2 --num_machines 1 --mixed_precision fp16 /nfshomes/asarkar6/aditya/PRISM/dreamengine/src/scripts/train/train.py \
#   --output_dir=$OUTPUT_DIR \
#   --pretrained_diffusion_ckpt="stabilityai/stable-diffusion-3.5-large" \
#   --pretrained_lmm_ckpt="Qwen/Qwen2-VL-2B-Instruct" \
#   --unfreeze_adapter \
#   --dataset_config=$DATASET_CONFIG \
#   --gradient_accumulation_steps=2 \
#   --cache_dir=$OUTPUT_DIR \
#   --train_batch_size=1 \
#   --learning_rate=5e-5 \
#   --max_grad_norm=1.0 \
#   --lr_scheduler="cosine" \
#   --lr_warmup_steps=50 \
#   --max_train_steps=1000 \
#   --seed="0" \
#   --checkpointing_steps=100 \
#   --lmm_output_layer_index=-1 \
#   --structure="direct" \
#   --cfg_ratio=0.1 \
#   --random_vit_skip \
  # --resume_from_checkpoint="/nfshomes/asarkar6/trinity/model_weights/DreamEngine-ObjectFusion/"

# run this part for stage 2 training.
# accelerate launch --num_processes 2 --num_machines 1 --mixed_precision bf16 --main_process_port 29501 --zero_stage 2 /nfshomes/asarkar6/aditya/PRISM/dreamengine/src/scripts/train/train.py \
#   --output_dir=$OUTPUT_DIR \
#   --pretrained_diffusion_ckpt="stabilityai/stable-diffusion-3.5-large" \
#   --pretrained_lmm_ckpt="Qwen/Qwen2-VL-2B-Instruct" \
#   --unfreeze_dit_layers="0,5,-5,-1" \
#   --unfreeze_adapter \
#   --lora_rank=32 \
#   --dataset_config=$DATASET_CONFIG \
#   --gradient_accumulation_steps=2 \
#   --cache_dir=$OUTPUT_DIR \
#   --train_batch_size=1 \
#   --learning_rate=5e-5 \
#   --max_grad_norm=1.0 \
#   --mixed_precision="bf16" \
#   --lr_scheduler="cosine" \
#   --lr_warmup_steps=500 \
#   --max_train_steps=1000 \
#   --seed="0" \
#   --checkpointing_steps=100 \
#   --lmm_output_layer_index=-1 \
#   --structure="direct" \
#   --cfg_ratio=0.1 \
#   --resume_from_checkpoint="/nfshomes/asarkar6/trinity/model_weights/dreamengine-checkpoint-100"

  # --unfreeze_adapter \
accelerate launch --main_process_port 29501 /nfshomes/asarkar6/aditya/PRISM/dreamengine/src/scripts/train/new_train.py \
  --output_dir=$OUTPUT_DIR \
  --pretrained_diffusion_ckpt="stabilityai/stable-diffusion-3.5-large" \
  --pretrained_lmm_ckpt="Qwen/Qwen2-VL-2B-Instruct" \
  --unfreeze_connector \
  --lora_rank=32 \
  --dataset_config=$DATASET_CONFIG \
  --gradient_accumulation_steps=2 \
  --train_batch_size=2 \
  --learning_rate=5e-5 \
  --max_grad_norm=1.0 \
  --mixed_precision="bf16" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=5000 \
  --max_train_steps=50000 \
  --validation_images=$VALIDATION_IMAGES \
  --validation_prompts="a lovely dog|a black and white flower, in the rain|a small  Robot looking at a small beautiful flower in the ground|a beautiful girl smiling and holding a flower" \
  --validation_steps=1000 \
  --seed="0" \
  --checkpointing_steps=10 \
  --lmm_output_layer_index=-1 \
  --structure="direct" \
  --cfg_ratio=0.1 \
  --num_latents=256