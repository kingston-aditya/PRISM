export DATASET_CONFIG="/nfshomes/asarkar6/aditya/DreamEngine/src/configs/ti2i_jdb1M.json"
export OUTPUT_DIR="/nfshomes/asarkar6/trinity/model_weights/"

export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes 2 --num_machines 1 --mixed_precision bf16 --zero_stage 2 train.py \
  --output_dir=$OUTPUT_DIR \
  --pretrained_diffusion_ckpt="stabilityai/stable-diffusion-3.5-large" \
  --pretrained_lmm_ckpt="Qwen/Qwen2-VL-2B-Instruct" \
  --unfreeze_adapter \
  --dataset_config=$DATASET_CONFIG \
  --gradient_accumulation_steps=2 \
  --cache_dir=$OUTPUT_DIR \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --max_grad_norm=1.0 \
  --mixed_precision="bf16" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=50 \
  --max_train_steps=1000 \
  --seed="0" \
  --checkpointing_steps=100 \
  --lmm_output_layer_index=-1 \
  --structure="direct" \
  --cfg_ratio=0.1 \
  --random_vit_skip


