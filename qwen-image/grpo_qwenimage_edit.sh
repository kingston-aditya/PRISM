# 8 GPU
accelerate launch --config_file ./config/deepspeed_zero3.yaml --num_processes=4 --main_process_port=19501 train_qwenimage_edit.py --config ./config/grpo.py:counting_qwenimage_edit_4gpu
