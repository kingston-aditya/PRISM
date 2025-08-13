config_file="/nfshomes/asarkar6/aditya/PRISM/metaquery/configs/qwen2p5vl3b_sd.yaml" 
run_name="metaquery_training"

OMP_NUM_THREADS=12 torchrun --nproc-per-node=2 --master-port=29501 /nfshomes/asarkar6/aditya/PRISM/metaquery/train.py \
 --run_name=$run_name \
 --config_file=$config_file \
 --base_dir="/nfshomes/asarkar6/scratch/test_image/" \
 --training_stage=2 \
 --resume_from_checkpoint="/nfshomes/asarkar6/trinity/model_weights/metaquery_training1/"