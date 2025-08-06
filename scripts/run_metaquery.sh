config_file="/home/saividyaranya/PRISM/configs/qwen2p5vl3b_sd.yaml" 
run_name="train"

OMP_NUM_THREADS=12 torchrun --nproc-per-node=8 --master-port=29501 /data/home/saividyaranya/PRISM/metaquery/train.py \
 --run_name=$run_name \
 --config_file=$config_file \
 --base_dir="/home/saividyaranya/PRISM/all_output_logs/mquery"