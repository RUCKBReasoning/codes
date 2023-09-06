import argparse
import os
import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_sft_dataset import SFTSQLGenerationDataset
from utils.db_utils import check_sql_executability, detect_special_char
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str)
    parser.add_argument('--dataset_path', type = str)
    parser.add_argument('--sic_path', type = str, default = None)
    
    parser.add_argument('--load_in_4bit', action = 'store_true')
    parser.add_argument('--load_in_8bit', action = 'store_true')

    opt = parser.parse_args()

    return opt

# TODO: refine this post processing function
def post_process(sql, schema_items):
    sql = sql.replace("\n", " ")
    for table in schema_items:
        for column_name in table["column_names"]:
            special_char_in_column_name = detect_special_char(column_name)
            if special_char_in_column_name and column_name in sql and "`"+column_name+"`" not in sql:
                sql = sql.replace(column_name, "`"+column_name+"`")
    sql = sql.replace(" order ", " `order` ")
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
    max_tokens = 2048
    max_new_tokens = 256

    tokenizer = AutoTokenizer.from_pretrained(opt.model_path)
    raw_dataset = json.load(open(opt.dataset_path))
    eval_set = SFTSQLGenerationDataset(
        opt.dataset_path,
        tokenizer,
        max_tokens - max_new_tokens,
        "eval",
        opt.sic_path
    )
    # TODO: current, we only support batch size = 1
    dataloader = DataLoader(eval_set, batch_size = 1)

    if opt.load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(opt.model_path, device_map = "auto", torch_dtype = torch.float16, load_in_4bit = True)
    elif opt.load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(opt.model_path, device_map = "auto", torch_dtype = torch.float16, load_in_8bit = True)
    else:
        model = AutoModelForCausalLM.from_pretrained(opt.model_path, device_map = "auto", torch_dtype = torch.float16)
    
    model.eval()

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

    with open("pred_sqls.txt", "w", encoding = 'utf-8') as f:
        for sql in predicted_sqls:
            f.write(sql + "\n")

    print("LLM name:", opt.model_path)
    if "bird" in opt.dataset_path:
        os.system('sh bird_evaluation/run_evaluation.sh')
    elif "spider" in opt.dataset_path:
        print("Execution accuracy:")
        os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider/dev_gold.sql --pred pred_sqls.txt --db ./data/sft_data_collections/spider/database --etype exec')
        print("Test suit execution accuracy:")
        os.system('python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider/dev_gold.sql --pred pred_sqls.txt --db test_suite_sql_eval/test_suite_database --etype exec')