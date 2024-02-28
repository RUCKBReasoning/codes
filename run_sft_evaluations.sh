set -e

# --------------- Spider dev --------------- #
# SFT CodeS-1B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-1b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-3B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-7B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-15B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-15b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256

# --------------- BIRD dev --------------- #
# SFT CodeS-1B on BIRD's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-1b-bird --sic_path ./sic_ckpts/sic_bird --table_num 6 --column_num 10 --dataset_path ./data/sft_bird_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-3B on BIRD's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-3b-bird --sic_path ./sic_ckpts/sic_bird --table_num 6 --column_num 10 --dataset_path ./data/sft_bird_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-7B on BIRD's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-7b-bird --sic_path ./sic_ckpts/sic_bird --table_num 6 --column_num 10 --dataset_path ./data/sft_bird_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-15B on BIRD's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-15b-bird --sic_path ./sic_ckpts/sic_bird --table_num 6 --column_num 10 --dataset_path ./data/sft_bird_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256

# --------------- BIRD dev w/ EK --------------- #
# SFT CodeS-1B on BIRD's training set (w/ EK)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-1b-bird-with-evidence --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 6 --column_num 10 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-3B on BIRD's training set (w/ EK)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-3b-bird-with-evidence --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 6 --column_num 10 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-7B on BIRD's training set (w/ EK)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-7b-bird-with-evidence --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 6 --column_num 10 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-15B on BIRD's training set (w/ EK)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-15b-bird-with-evidence --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 6 --column_num 10 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256

# --------------- Spider-Syn --------------- #
# SFT CodeS-1B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-1b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_syn_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-3B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_syn_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-7B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_syn_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-15B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-15b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_syn_text2sql.json --max_tokens 4096 --max_new_tokens 256

# --------------- Spider-Realistic --------------- #
# SFT CodeS-1B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-1b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_realistic_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-3B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_realistic_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-7B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_realistic_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-15B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-15b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_realistic_text2sql.json --max_tokens 4096 --max_new_tokens 256

# --------------- Spider-DK --------------- #
# SFT CodeS-1B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-1b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_dk_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-3B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_dk_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-7B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_dk_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-15B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-15b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_spider_dk_text2sql.json --max_tokens 4096 --max_new_tokens 256

# --------------- Dr.Spider --------------- #
# SFT CodeS-1B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-1b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_dr_spider_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-3B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-3b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_dr_spider_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-7B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-7b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_dr_spider_text2sql.json --max_tokens 4096 --max_new_tokens 256

# SFT CodeS-15B on Spider's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-15b-spider --sic_path ./sic_ckpts/sic_spider --table_num 6 --column_num 10 --dataset_path ./data/sft_dr_spider_text2sql.json --max_tokens 4096 --max_new_tokens 256

# --------------- Bank-Financials --------------- #
# SFT CodeS-7B on Bank-Financials's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-7b-bank --sic_path ./sic_ckpts/sic_bird --table_num 6 --column_num 10 --dataset_path ./data/sft_bank_financials_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256
# SFT CodeS-7B on merged training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-7b-merged --sic_path ./sic_ckpts/sic_bird --table_num 6 --column_num 10 --dataset_path ./data/sft_bank_financials_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256

# --------------- Aminer-Simplified --------------- #
# SFT CodeS-7B on Aminer-Simplified's training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-7b-aminer --sic_path ./sic_ckpts/sic_bird --table_num 6 --column_num 10 --dataset_path ./data/sft_aminer_simplified_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256
# SFT CodeS-7B on merged training set
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --llm_path seeklhy/codes-7b-merged --sic_path ./sic_ckpts/sic_bird --table_num 6 --column_num 10 --dataset_path ./data/sft_aminer_simplified_dev_text2sql.json --max_tokens 4096 --max_new_tokens 256