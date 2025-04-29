python /nfshomes/asarkar6/aditya/PRISM/dataset/synthetic/run_llm.py \
 --start_len 0 \
 --end_len 750000 \
 --job_id 0

accelerate launch /nfshomes/asarkar6/aditya/PRISM/dataset/synthetic/run_final_syn.py \
 --start_len 0 \
 --end_len 750000 \
 --job_id 0
