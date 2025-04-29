python /nfshomes/asarkar6/aditya/PRISM/dataset/real/datagen_init.py \
 --start_len 0 \
 --end_len 750000 \
 --job_id 0

python /nfshomes/asarkar6/aditya/PRISM/dataset/real/run_llm.py \
 --start_len 0 \
 --end_len 750000 \
 --job_id 0

accelerate launch /nfshomes/asarkar6/aditya/PRISM/dataset/real/datagen_launch.py \
 --start_len 0 \
 --end_len 750000 \
 --job_id 0