export DATASET_CONFIG="/bucket/YamadaU/asarkar/CC3M/"
export OUTPUT_DIR="/work/YamadaU/asarkar/prism-outputs/weights/"

export TOKENIZERS_PARALLELISM=false

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

accelerate launch --main_process_port 29501 /work/YamadaU/asarkar/PRISM/dreamengine/src/scripts/train/new_train.py \
  --output_dir=$OUTPUT_DIR \
  --pretrained_diffusion_ckpt="stabilityai/stable-diffusion-3.5-large" \
  --pretrained_lmm_ckpt="Qwen/Qwen2-VL-2B-Instruct" \
  --unfreeze_connector \
  --lora_rank=32 \
  --dataset_dir=$DATASET_CONFIG \
  --gradient_accumulation_steps=2 \
  --train_batch_size=4 \
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
  --checkpointing_steps=1000 \
  --lmm_output_layer_index=-1 \
  --structure="direct" \
  --cfg_ratio=0.1 \
  --num_latents=256 \
  --resume_from_checkpoint="/work/YamadaU/asarkar/prism-outputs/weights/DreamEngine-ObjectFusion/transformer/"
