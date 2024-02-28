set -e

# --------------- Few-shot CodeS-1B --------------- #
# Spider's dev (1-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-1b --sic_path ./sic_ckpts/sic_spider --table_num 5 --column_num 6 --dataset_path ./data/sft_spider_dev_text2sql.json --demonstration_set_path ./data/sft_spider_train_text2sql.json --num_of_demonstrations 1 --max_tokens 8192 --max_new_tokens 256

# Spider's dev (3-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-1b --sic_path ./sic_ckpts/sic_spider --table_num 5 --column_num 6 --dataset_path ./data/sft_spider_dev_text2sql.json --demonstration_set_path ./data/sft_spider_train_text2sql.json --num_of_demonstrations 3 --max_tokens 8192 --max_new_tokens 256

# Spider's dev (5-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-1b --sic_path ./sic_ckpts/sic_spider --table_num 5 --column_num 6 --dataset_path ./data/sft_spider_dev_text2sql.json --demonstration_set_path ./data/sft_spider_train_text2sql.json --num_of_demonstrations 5 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev (1-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-1b --sic_path ./sic_ckpts/sic_bird --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_dev_text2sql.json --demonstration_set_path ./data/sft_bird_train_text2sql.json --num_of_demonstrations 1 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev (3-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-1b --sic_path ./sic_ckpts/sic_bird --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_dev_text2sql.json --demonstration_set_path ./data/sft_bird_train_text2sql.json --num_of_demonstrations 3 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev (5-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-1b --sic_path ./sic_ckpts/sic_bird --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_dev_text2sql.json --demonstration_set_path ./data/sft_bird_train_text2sql.json --num_of_demonstrations 5 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev w/ EK (1-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-1b --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --demonstration_set_path ./data/sft_bird_with_evidence_train_text2sql.json --num_of_demonstrations 1 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev w/ EK (3-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-1b --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --demonstration_set_path ./data/sft_bird_with_evidence_train_text2sql.json --num_of_demonstrations 3 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev w/ EK (5-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-1b --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --demonstration_set_path ./data/sft_bird_with_evidence_train_text2sql.json --num_of_demonstrations 5 --max_tokens 8192 --max_new_tokens 256

# --------------- Few-shot CodeS-3B --------------- #
# Spider's dev (1-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-3b --sic_path ./sic_ckpts/sic_spider --table_num 5 --column_num 6 --dataset_path ./data/sft_spider_dev_text2sql.json --demonstration_set_path ./data/sft_spider_train_text2sql.json --num_of_demonstrations 1 --max_tokens 8192 --max_new_tokens 256

# Spider's dev (3-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-3b --sic_path ./sic_ckpts/sic_spider --table_num 5 --column_num 6 --dataset_path ./data/sft_spider_dev_text2sql.json --demonstration_set_path ./data/sft_spider_train_text2sql.json --num_of_demonstrations 3 --max_tokens 8192 --max_new_tokens 256

# Spider's dev (5-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-3b --sic_path ./sic_ckpts/sic_spider --table_num 5 --column_num 6 --dataset_path ./data/sft_spider_dev_text2sql.json --demonstration_set_path ./data/sft_spider_train_text2sql.json --num_of_demonstrations 5 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev (1-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-3b --sic_path ./sic_ckpts/sic_bird --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_dev_text2sql.json --demonstration_set_path ./data/sft_bird_train_text2sql.json --num_of_demonstrations 1 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev (3-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-3b --sic_path ./sic_ckpts/sic_bird --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_dev_text2sql.json --demonstration_set_path ./data/sft_bird_train_text2sql.json --num_of_demonstrations 3 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev (5-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-3b --sic_path ./sic_ckpts/sic_bird --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_dev_text2sql.json --demonstration_set_path ./data/sft_bird_train_text2sql.json --num_of_demonstrations 5 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev w/ EK (1-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-3b --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --demonstration_set_path ./data/sft_bird_with_evidence_train_text2sql.json --num_of_demonstrations 1 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev w/ EK (3-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-3b --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --demonstration_set_path ./data/sft_bird_with_evidence_train_text2sql.json --num_of_demonstrations 3 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev w/ EK (5-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-3b --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --demonstration_set_path ./data/sft_bird_with_evidence_train_text2sql.json --num_of_demonstrations 5 --max_tokens 8192 --max_new_tokens 256


# --------------- Few-shot CodeS-7B --------------- #
# Spider's dev (1-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-7b --sic_path ./sic_ckpts/sic_spider --table_num 5 --column_num 6 --dataset_path ./data/sft_spider_dev_text2sql.json --demonstration_set_path ./data/sft_spider_train_text2sql.json --num_of_demonstrations 1 --max_tokens 8192 --max_new_tokens 256

# Spider's dev (3-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-7b --sic_path ./sic_ckpts/sic_spider --table_num 5 --column_num 6 --dataset_path ./data/sft_spider_dev_text2sql.json --demonstration_set_path ./data/sft_spider_train_text2sql.json --num_of_demonstrations 3 --max_tokens 8192 --max_new_tokens 256

# Spider's dev (5-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-7b --sic_path ./sic_ckpts/sic_spider --table_num 5 --column_num 6 --dataset_path ./data/sft_spider_dev_text2sql.json --demonstration_set_path ./data/sft_spider_train_text2sql.json --num_of_demonstrations 5 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev (1-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-7b --sic_path ./sic_ckpts/sic_bird --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_dev_text2sql.json --demonstration_set_path ./data/sft_bird_train_text2sql.json --num_of_demonstrations 1 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev (3-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-7b --sic_path ./sic_ckpts/sic_bird --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_dev_text2sql.json --demonstration_set_path ./data/sft_bird_train_text2sql.json --num_of_demonstrations 3 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev (5-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-7b --sic_path ./sic_ckpts/sic_bird --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_dev_text2sql.json --demonstration_set_path ./data/sft_bird_train_text2sql.json --num_of_demonstrations 5 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev w/ EK (1-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-7b --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --demonstration_set_path ./data/sft_bird_with_evidence_train_text2sql.json --num_of_demonstrations 1 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev w/ EK (3-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-7b --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --demonstration_set_path ./data/sft_bird_with_evidence_train_text2sql.json --num_of_demonstrations 3 --max_tokens 8192 --max_new_tokens 256

# BIRD's dev w/ EK (5-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-7b --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --demonstration_set_path ./data/sft_bird_with_evidence_train_text2sql.json --num_of_demonstrations 5 --max_tokens 8192 --max_new_tokens 256


# --------------- Few-shot CodeS-15B --------------- #
# Spider's dev (1-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-15b --sic_path ./sic_ckpts/sic_spider --table_num 5 --column_num 6 --dataset_path ./data/sft_spider_dev_text2sql.json --demonstration_set_path ./data/sft_spider_train_text2sql.json --num_of_demonstrations 1 --max_tokens 6144 --max_new_tokens 256

# Spider's dev (3-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-15b --sic_path ./sic_ckpts/sic_spider --table_num 5 --column_num 6 --dataset_path ./data/sft_spider_dev_text2sql.json --demonstration_set_path ./data/sft_spider_train_text2sql.json --num_of_demonstrations 3 --max_tokens 6144 --max_new_tokens 256

# Spider's dev (5-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-15b --sic_path ./sic_ckpts/sic_spider --table_num 5 --column_num 6 --dataset_path ./data/sft_spider_dev_text2sql.json --demonstration_set_path ./data/sft_spider_train_text2sql.json --num_of_demonstrations 5 --max_tokens 6144 --max_new_tokens 256

# BIRD's dev (1-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-15b --sic_path ./sic_ckpts/sic_bird --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_dev_text2sql.json --demonstration_set_path ./data/sft_bird_train_text2sql.json --num_of_demonstrations 1 --max_tokens 6144 --max_new_tokens 256

# BIRD's dev (3-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-15b --sic_path ./sic_ckpts/sic_bird --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_dev_text2sql.json --demonstration_set_path ./data/sft_bird_train_text2sql.json --num_of_demonstrations 3 --max_tokens 6144 --max_new_tokens 256

# BIRD's dev (5-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-15b --sic_path ./sic_ckpts/sic_bird --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_dev_text2sql.json --demonstration_set_path ./data/sft_bird_train_text2sql.json --num_of_demonstrations 5 --max_tokens 6144 --max_new_tokens 256

# BIRD's dev w/ EK (1-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-15b --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --demonstration_set_path ./data/sft_bird_with_evidence_train_text2sql.json --num_of_demonstrations 1 --max_tokens 6144 --max_new_tokens 256

# BIRD's dev w/ EK (3-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-15b --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --demonstration_set_path ./data/sft_bird_with_evidence_train_text2sql.json --num_of_demonstrations 3 --max_tokens 6144 --max_new_tokens 256

# BIRD's dev w/ EK (5-shot)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --llm_path seeklhy/codes-15b --sic_path ./sic_ckpts/sic_bird_with_evidence --table_num 5 --column_num 6 --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json --demonstration_set_path ./data/sft_bird_with_evidence_train_text2sql.json --num_of_demonstrations 5 --max_tokens 6144 --max_new_tokens 256
