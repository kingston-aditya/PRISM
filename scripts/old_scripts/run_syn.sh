start_len=75_000
end_len=150_000
job_id=2

python /home/saividyaranya/PRISM/dataset/synthetic/run_llm.py \
 --start_len $start_len \
 --end_len $end_len \
 --job_id $job_id

# accelerate launch /nfshomes/asarkar6/aditya/PRISM/dataset/synthetic/run_final_syn.py \
#  --start_len $start_len \
#  --end_len $end_len \
#  --job_id $job_id
