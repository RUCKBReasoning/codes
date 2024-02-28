import argparse
import os
import torch
import json
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_sft_dataset import SFTSQLGenerationDataset
from utils.db_utils import check_sql_executability, detect_special_char
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_path', type = str)
    parser.add_argument('--sic_path', type = str)
    parser.add_argument('--table_num', type = int, default = 6)
    parser.add_argument('--column_num', type = int, default = 10)

    parser.add_argument('--dataset_path', type = str)

    parser.add_argument('--max_tokens', type = int, default = 4096)
    parser.add_argument('--max_new_tokens', type = int, default = 256)
    
    opt = parser.parse_args()

    return opt

def post_process(sql, schema_items):
    sql = sql.replace("\n", " ")
    for table in schema_items:
        for column_name in table["column_names"]:
            if detect_special_char(column_name) and column_name in sql:
                sql = sql.replace(column_name, "`"+column_name+"`")

    while "``" in sql:
        sql = sql.replace("``", "`")

    return sql

def text2sql_func(model, inputs, tokenizer, max_new_tokens):
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            num_beams = 4,
            num_return_sequences = 4
        )

    # print(tokenizer.decode(generate_ids[0]))
    generated_sqls = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = True, clean_up_tokenization_spaces = False)
    # print(generated_sqls)

    return generated_sqls

if __name__ == "__main__":
    opt = parse_option()
    print(opt)
    max_tokens = opt.max_tokens
    max_new_tokens = opt.max_new_tokens

    tokenizer = AutoTokenizer.from_pretrained(opt.llm_path)
    raw_dataset = json.load(open(opt.dataset_path))
    eval_set = SFTSQLGenerationDataset(
        opt.dataset_path,
        tokenizer,
        max_tokens - max_new_tokens,
        "eval",
        opt.table_num,
        opt.column_num,
        opt.sic_path
    )

    # TODO: current, we only support batch size = 1
    dataloader = DataLoader(eval_set, batch_size = 1)
    model = AutoModelForCausalLM.from_pretrained(opt.llm_path, device_map = "auto", torch_dtype = torch.float16)
    
    model.eval()
    start_time = time.time()
    predicted_sqls = []
    for raw_data, batch_data in tqdm(zip(raw_dataset, dataloader)):
        for key in batch_data:
            batch_data[key] = batch_data[key].to(model.device)
        generated_sqls = text2sql_func(model, batch_data, tokenizer, max_new_tokens)
        generated_sqls = [post_process(generated_sql, raw_data["schema"]["schema_items"]) for generated_sql in generated_sqls]

        final_generated_sql = None
        for generated_sql in generated_sqls:
            execution_error = check_sql_executability(generated_sql, raw_data["db_path"])
            if execution_error is None: # the generated sql has no execution errors, we will return it as the final generated sql
                final_generated_sql = generated_sql
                break

        if final_generated_sql is None:
            if generated_sqls[0].strip() != "":
                final_generated_sql = generated_sqls[0]
            else:
                final_generated_sql = "SQL placeholder"
        
        print(final_generated_sql)
        predicted_sqls.append(final_generated_sql)
    end_time = time.time()
    print("LLM name: {} | Total time: {}s | Example number: {} | Average time: {}s".format(
        opt.llm_path, 
        end_time - start_time,
        len(raw_dataset),
        (end_time - start_time) / len(raw_dataset)
        )
    )

    print("LLM name:", opt.llm_path)
    if "bird" in opt.dataset_path:
        bird_results_dict = dict()
        for idx, (data, predicted_sql) in enumerate(zip(raw_dataset, predicted_sqls)):
            bird_results_dict[idx] = predicted_sql + "\t----- bird -----\t" + data["db_id"]
        with open("predict_dev.json", "w", encoding = 'utf-8') as f:
            f.write(json.dumps(bird_results_dict, indent = 2, ensure_ascii = False))
        os.system("sh bird_evaluation/run_evaluation.sh predict_dev.json")
    elif "spider_dev" in opt.dataset_path:
        with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        print("Execution accuracy:")
        os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider/dev_gold.sql --pred pred_sqls.txt --db ./data/sft_data_collections/spider/database --etype exec')
        with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        print("Test suit execution accuracy:")
        os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider/dev_gold.sql --pred pred_sqls.txt --db test_suite_sql_eval/test_suite_database --etype exec')
    elif "spider_dk" in opt.dataset_path:
        with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        print("Execution accuracy:")
        os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/Spider-DK/dk_gold.sql --pred pred_sqls.txt --db ./data/sft_data_collections/spider/database --etype exec')
    elif "spider_realistic" in opt.dataset_path:
        with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        print("Execution accuracy:")
        os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider-realistic/realistic_gold.sql --pred pred_sqls.txt --db ./data/sft_data_collections/spider/database --etype exec')
        with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        print("Test suit execution accuracy:")
        os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider-realistic/realistic_gold.sql --pred pred_sqls.txt --db test_suite_sql_eval/test_suite_database --etype exec')
    elif "spider_syn" in opt.dataset_path:
        with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        print("Execution accuracy:")
        os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/Spider-Syn/Spider-Syn/syn_dev_gold.sql --pred pred_sqls.txt --db ./data/sft_data_collections/spider/database --etype exec')
        with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        print("Test suit execution accuracy:")
        os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/Spider-Syn/Spider-Syn/syn_dev_gold.sql --pred pred_sqls.txt --db test_suite_sql_eval/test_suite_database --etype exec')
    elif "dr_spider" in opt.dataset_path:
        test_set_names = ['NLQ_column_value', 'SQL_DB_text', 'DB_schema_abbreviation', 'SQL_DB_number', \
            'SQL_comparison', 'NLQ_column_carrier', 'NLQ_multitype', 'NLQ_others', 'NLQ_keyword_synonym', \
                'SQL_NonDB_number', 'DB_schema_synonym', 'NLQ_column_synonym', 'DB_DBcontent_equivalence', \
                    'NLQ_keyword_carrier', 'NLQ_value_synonym', 'NLQ_column_attribute', 'SQL_sort_order']
        
        for test_set_name in test_set_names:
            print(test_set_name)
            test_set_predicted_sqls = [predicted_sql for predicted_sql, raw_data in zip(predicted_sqls, raw_dataset) if raw_data["source"] == "dr.spider-" + test_set_name]

            database_file_path = "database_post_perturbation" if test_set_name.startswith("DB_") else "databases"
            db_path = os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, database_file_path)
            gold_path = os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, "gold_post_perturbation.sql")

            with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
                for sql in test_set_predicted_sqls:
                    f.write(sql + "\n")
            print("Execution accuracy:")
            os.system('python -u test_suite_sql_eval/evaluation.py --gold {} --pred pred_sqls.txt --db {} --etype exec'.format(gold_path, db_path))
    elif "bank" in opt.dataset_path:
        with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        print("Execution accuracy:")
        os.system('python -u evaluate_ex.py --pred pred_sqls.txt --gold {} --db ./data/sft_data_collections/domain_datasets/databases/Bank_Financials/Bank_Financials.sqlite'.format(opt.dataset_path))
    elif "aminer" in opt.dataset_path:
        with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
            for sql in predicted_sqls:
                f.write(sql + "\n")
        print("Execution accuracy:")
        os.system('python -u evaluate_ex.py --pred pred_sqls.txt --gold {} --db ./data/sft_data_collections/domain_datasets/databases/Aminer_Simplified/Aminer_Simplified.sqlite'.format(opt.dataset_path))