from tqdm import tqdm
from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer
import argparse

def prepare(tokenizer, pt_corpus_dir):
    # set `model_max_length` to a very large integer to avoid warning
    tokenizer.model_max_length = int(1e30)

    num_proc = 32
    pure_sql_dataset = Dataset.from_json("./codes_pretrain_corpus/pure_sql.jsonl")
    text2code_dataset = Dataset.from_json("./codes_pretrain_corpus/text2code.jsonl")
    text2text_dataset = Dataset.from_json("./codes_pretrain_corpus/text2text.jsonl")

    print(pure_sql_dataset)
    print(text2code_dataset)
    print(text2text_dataset)

    def tokenize(sequences):
        input_ids = tokenizer(sequences, truncation = False)["input_ids"]
        # add EOS token
        input_ids = [ids + [tokenizer.eos_token_id] for ids in input_ids]
        length = [len(ids) for ids in input_ids]

        return {'input_ids': input_ids, 'length': length}

    def process_sql_corpus(examples):
        sequences = [sql for sql in examples["sql"]]
        sequences = [seq.strip() for seq in sequences]
        return tokenize(sequences)
    
    def process_text2code(examples):
        sequences = [text + "\n" + code for text, code in zip(examples["text"], examples["code"])]
        sequences = [sequence.strip() for sequence in sequences]
        return tokenize(sequences)

    def process_text2text(examples):
        sequences = [input_text + "\n" + output_text for input_text, output_text in zip(examples["input_text"], examples["output_text"])]
        sequences = [sequence.strip() for sequence in sequences]
        return tokenize(sequences)

    pure_sql_dataset = pure_sql_dataset.map(
        process_sql_corpus, 
        num_proc = num_proc, 
        desc = "tokenizing the sql only dataset.", 
        remove_columns = ["sql"],
        batched = True
    )
    text2code_dataset = text2code_dataset.map(
        process_text2code, 
        num_proc = num_proc, 
        desc = "tokenizing the text2code dataset.", 
        remove_columns = ["text", "code"],
        batched = True
    )
    text2text_dataset = text2text_dataset.map(
        process_text2text, 
        num_proc = num_proc, 
        desc = "tokenizing the text2text dataset.", 
        remove_columns = ["input_text", "output_text"],
        batched = True
    )

    print(pure_sql_dataset)
    print(text2code_dataset)
    print(text2text_dataset)

    final_copurs = [pure_sql_dataset, pure_sql_dataset, text2code_dataset, text2text_dataset]
    
    arr_len = sum(np.sum(tokenized_dataset['length']) for tokenized_dataset in final_copurs)
    print("There are {} tokens in the corpus".format(arr_len))
    dtype = np.uint16 # (can do since starcoder's vocab size == 49152 is < 2**16 == 65536)
    arr = np.memmap(pt_corpus_dir, dtype = dtype, mode = 'w+', shape = (arr_len,))
    idx = 0
    total_batches = 2048
    
    # concatenate all the ids in each dataset into one large file
    for tokenized_dataset in final_copurs:
        for batch_idx in tqdm(range(total_batches)):
            # Batch together samples for faster write
            batch = tokenized_dataset.shard(
                num_shards = total_batches, 
                index = batch_idx, 
                contiguous = True).with_format('numpy')
            arr_batch = np.concatenate(batch['input_ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")
    prepare(tokenizer, "tokenized_corpus.bin")