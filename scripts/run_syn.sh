start_len = 0
end_len = 75000
job_id = 0

python /nfshomes/asarkar6/aditya/PRISM/dataset/synthetic/run_llm.py \
 --start_len $start_len \
 --end_len $end_len \
 --job_id $job_id

accelerate launch /nfshomes/asarkar6/aditya/PRISM/dataset/synthetic/run_final_syn.py \
 --start_len $start_len \
 --end_len $end_len \
 --job_id $job_id
