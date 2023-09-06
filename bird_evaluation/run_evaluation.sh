db_root_path='./data/sft_data_collections/bird/dev/dev_databases/'
data_mode='dev'
diff_json_path='./data/sft_data_collections/bird/dev/dev.json'
predicted_sql_path_kg='./bird_evaluation/'
ground_truth_path='./data/sft_data_collections/bird/dev/'
num_cpus=16
meta_time_out=30.0
mode_gt='gt'
mode_predict='gpt'

echo '''reformat predicted results'''
python -u ./bird_evaluation/extract_results.py

echo '''starting to compare with knowledge for ex'''
python3 -u ./bird_evaluation/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

echo '''starting to compare with knowledge for ves'''
python3 -u ./bird_evaluation/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}