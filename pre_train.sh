set -e

# Pre-train CodeS-1B
accelerate launch train_causal_lm.py --per_device_train_batch_size 2 --block_size 8192 --seed 42 --pretrained_model_name_or_path bigcode/starcoderbase-1b --epochs 1 --lr 5e-5 --warmup_ratio 0.0 --checkpointing_steps 500 --tensorboard_log_dir ./train_logs/pt-codes-1b --mode pt --output_ckpt_dir ./ckpts/pt-codes-1b --pt_data_dir ./tokenized_corpus.bin
# Pre-train CodeS-3B
accelerate launch train_causal_lm.py --per_device_train_batch_size 2 --block_size 8192 --seed 42 --pretrained_model_name_or_path bigcode/starcoderbase-3b --epochs 1 --lr 5e-5 --warmup_ratio 0.0 --checkpointing_steps 500 --tensorboard_log_dir ./train_logs/pt-codes-3b --mode pt --output_ckpt_dir ./ckpts/pt-codes-3b --pt_data_dir ./tokenized_corpus.bin
# Pre-train CodeS-7B
accelerate launch train_causal_lm.py --per_device_train_batch_size 2 --block_size 8192 --seed 42 --pretrained_model_name_or_path bigcode/starcoderbase-7b --epochs 1 --lr 5e-5 --warmup_ratio 0.0 --checkpointing_steps 500 --tensorboard_log_dir ./train_logs/pt-codes-7b --mode pt --output_ckpt_dir ./ckpts/pt-codes-7b --pt_data_dir ./tokenized_corpus.bin
# Pre-train CodeS-15B
accelerate launch train_causal_lm.py --per_device_train_batch_size 2 --block_size 8192 --seed 42 --pretrained_model_name_or_path bigcode/starcoder --epochs 1 --lr 5e-5 --warmup_ratio 0.0 --checkpointing_steps 500 --tensorboard_log_dir ./train_logs/pt-codes-15b --mode pt --output_ckpt_dir ./ckpts/pt-codes-15b --pt_data_dir ./tokenized_corpus.bin