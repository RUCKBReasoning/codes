set -e

# Train schema filter using Spider
python -u train_schema_item_filter.py \
    --batch_size 4 \
    --gradient_descent_step 8 \
    --device 0 \
    --learning_rate 1e-5 \
    --gamma 2.0 \
    --alpha 0.75 \
    --epochs 64 \
    --patience 8 \
    --seed 42 \
    --save_path ./sic_ckpts/sic_spider \
    --tensorboard_save_path ./train_logs/sic_spider \
    --train_filepath ./data/sft_spider_train_text2sql.json \
    --dev_filepath ./data/sft_spider_dev_text2sql.json \
    --model_name_or_path roberta-large \
    --mode train

# Train schema filter using BIRD
python -u train_schema_item_filter.py \
    --batch_size 4 \
    --gradient_descent_step 8 \
    --device 0 \
    --learning_rate 1e-5 \
    --gamma 2.0 \
    --alpha 0.75 \
    --epochs 64 \
    --patience 8 \
    --seed 42 \
    --save_path ./sic_ckpts/sic_bird \
    --tensorboard_save_path ./train_logs/sic_bird \
    --train_filepath ./data/sft_bird_train_text2sql.json \
    --dev_filepath ./data/sft_bird_dev_text2sql.json \
    --model_name_or_path roberta-large \
    --mode train

# Train schema filter using BIRD with external knowledge
python -u train_schema_item_filter.py \
    --batch_size 4 \
    --gradient_descent_step 8 \
    --device 0 \
    --learning_rate 1e-5 \
    --gamma 2.0 \
    --alpha 0.75 \
    --epochs 64 \
    --patience 8 \
    --seed 42 \
    --save_path ./sic_ckpts/sic_bird_with_evidence \
    --tensorboard_save_path ./train_logs/sic_bird_with_evidence \
    --train_filepath ./data/sft_bird_with_evidence_train_text2sql.json \
    --dev_filepath ./data/sft_bird_with_evidence_dev_text2sql.json \
    --model_name_or_path roberta-large \
    --mode train