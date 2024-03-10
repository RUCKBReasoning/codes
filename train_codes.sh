set -e

# -------------- Training on Spider -------------- #
# SFT CodeS-1B on Spider
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-1b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-1b-spider --mode sft --output_ckpt_dir ./ckpts/codes-1b-spider --text2sql_data_dir ./data/sft_spider_train_text2sql.json --table_num 6 --column_num 10
# SFT CodeS-3B on Spider
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-3b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-3b-spider --mode sft --output_ckpt_dir ./ckpts/codes-3b-spider --text2sql_data_dir ./data/sft_spider_train_text2sql.json --table_num 6 --column_num 10
# SFT CodeS-7B on Spider
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-7b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-7b-spider --mode sft --output_ckpt_dir ./ckpts/codes-7b-spider --text2sql_data_dir ./data/sft_spider_train_text2sql.json --table_num 6 --column_num 10
# SFT CodeS-15B on Spider
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-15b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-15b-spider --mode sft --output_ckpt_dir ./ckpts/codes-15b-spider --text2sql_data_dir ./data/sft_spider_train_text2sql.json --table_num 6 --column_num 10

# -------------- Training on BIRD -------------- #
# SFT CodeS-1B on BIRD
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-1b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-1b-bird --mode sft --output_ckpt_dir ./ckpts/codes-1b-bird --text2sql_data_dir ./data/sft_bird_train_text2sql.json --table_num 6 --column_num 10
# SFT CodeS-3B on BIRD
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-3b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-3b-bird --mode sft --output_ckpt_dir ./ckpts/codes-3b-bird --text2sql_data_dir ./data/sft_bird_train_text2sql.json --table_num 6 --column_num 10
# SFT CodeS-7B on BIRD
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-7b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-7b-bird --mode sft --output_ckpt_dir ./ckpts/codes-7b-bird --text2sql_data_dir ./data/sft_bird_train_text2sql.json --table_num 6 --column_num 10
# SFT CodeS-15B on BIRD
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-15b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-15b-bird --mode sft --output_ckpt_dir ./ckpts/codes-15b-bird --text2sql_data_dir ./data/sft_bird_train_text2sql.json --table_num 6 --column_num 10

# -------------- Training on BIRD with external knowledge -------------- #
# SFT CodeS-1B on BIRD with external knowledge
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-1b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-1b-bird-with-evidence --mode sft --output_ckpt_dir ./ckpts/codes-1b-bird-with-evidence --text2sql_data_dir ./data/sft_bird_with_evidence_train_text2sql.json --table_num 6 --column_num 10
# SFT CodeS-3B on BIRD with external knowledge
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-3b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-3b-bird-with-evidence --mode sft --output_ckpt_dir ./ckpts/codes-3b-bird-with-evidence --text2sql_data_dir ./data/sft_bird_with_evidence_train_text2sql.json --table_num 6 --column_num 10
# SFT CodeS-7B on BIRD with external knowledge
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-7b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-7b-bird-with-evidence --mode sft --output_ckpt_dir ./ckpts/codes-7b-bird-with-evidence --text2sql_data_dir ./data/sft_bird_with_evidence_train_text2sql.json --table_num 6 --column_num 10
# SFT CodeS-15B on BIRD with external knowledge
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-15b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-15b-bird-with-evidence --mode sft --output_ckpt_dir ./ckpts/codes-15b-bird-with-evidence --text2sql_data_dir ./data/sft_bird_with_evidence_train_text2sql.json --table_num 6 --column_num 10

# -------------- Training on our augmented datasets (i.e., Bank-Financials and Aminer-Simplified) -------------- #
# SFT CodeS-7B on Bank-Financials
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-7b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-7b-bank --mode sft --output_ckpt_dir ./ckpts/codes-7b-bank --text2sql_data_dir ./data/sft_bank_financials_train_text2sql.json --table_num 6 --column_num 10
# SFT CodeS-7B on Aminer-Simplified
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-7b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-7b-aminer --mode sft --output_ckpt_dir ./ckpts/codes-7b-aminer --text2sql_data_dir ./data/sft_aminer_simplified_train_text2sql.json --table_num 6 --column_num 10

# -------------- Training on all-merged dataset (i.e., Spider + BIRD w/ external knowledge + Bank-Financials + Aminer-Simplified) -------------- #
# SFT CodeS-7B on all-merged dataset
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 4096 --seed 42 --pretrained_model_name_or_path seeklhy/codes-7b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-7b-merged --mode sft --output_ckpt_dir ./ckpts/codes-7b-merged --text2sql_data_dir ./data/sft_all_merged_train_text2sql.json --table_num 6 --column_num 10
