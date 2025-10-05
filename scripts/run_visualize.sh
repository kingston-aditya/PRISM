accelerate launch --mixed_precision fp16 --main_process_port 29501 /nfshomes/asarkar6/aditya/PRISM/dataset/filter.py \
 --pretrained_model_name_or_path="laion/CLIP-ViT-g-14-laion2B-s12B-b42K" \
 --output_dir="/nfshomes/asarkar6/aditya/" \
 --cache_dir="/nfshomes/asarkar6/trinity/model_weights/" \
 --make_plot="true" \
 --train_batch_size=2 \
 --mixed_precision="fp16" 
