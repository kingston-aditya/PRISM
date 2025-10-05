# 8 GPU
torchrun --standalone --nproc_per_node=2 --master_port=19501 train_sd15_edit.py --config config/grpo.py:counting_sd15_edit_2gpu