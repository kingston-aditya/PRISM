start_len=1_500_000
end_len=2_250_000
job_id=3

# python /nfshomes/asarkar6/aditya/PRISM/dataset/real/datagen_init.py \
#  --start_len $start_len \
#  --end_len $end_len \
#  --job_id $job_id

# python /nfshomes/asarkar6/aditya/PRISM/dataset/real/run_llm.py \
#  --start_len $start_len \
#  --end_len $end_len \
#   --job_id $job_id

# python /data/home/saividyaranya/PRISM/dataset/real/datagen_final.py \
accelerate launch /data/home/saividyaranya/PRISM/dataset/real/datagen_final.py \
 --start_len $start_len \
 --end_len $end_len \
 --job_id $job_id
