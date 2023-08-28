db_root_path='./data/sft_data_collections/bird/dev/dev_databases'
data_mode='dev'
predicted_sql_path_kg='./bird/llm/exp_result/my_model_output_kg/'
ground_truth_path='./data/sft_data_collections/bird/dev/'
num_cpus=16
time_out=60
mode_gt='gt'
mode_predict='gpt'

echo '''starting to compare with knowledge'''
python3 -u ./src/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --time_out ${time_out} --mode_gt ${mode_gt} --mode_predict ${mode_predict}