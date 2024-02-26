# CodeS: An open pre-trained large language model (LLM) for SQL generation

We release **CodeS**, a series of **Code** LLMs specifically optimized for **S**QL generation. CodeS is **incrementally pre-trained** based on StarCoder using a large SQL-related corpus. The CodeS series encompasses four distinct scales: [CodeS-1B](https://huggingface.co/seeklhy/codes-1b), [CodeS-3B](https://huggingface.co/seeklhy/codes-3b), [CodeS-7B](https://huggingface.co/seeklhy/codes-7b), and [CodeS-15B](https://huggingface.co/seeklhy/codes-15b). CodeS-1B, 3B, and 7B are based on StarCoderBase-1B, 3B, and 7B and support the max length of 8,192. Meanwhile, CodeS-15B, derived from StarCoder-15B, accommodates sequences of up to 6,144 tokens. 

In addition, We've launched a text-to-SQL demo based on CodeS. You can access it at [RUCKBReasoning/text2sql-demo](https://github.com/RUCKBReasoning/text2sql-demo). Feel free to give it a try and follow the provided instructions to create your own custom demo tailored to your databases!

TODO:
- Release fine-tuning code ⚪
- Release data pre-processing code ⚪

## Evaluation Results
We evaluate CodeS on two challenging Text-to-SQL benchmarks: Spider and BIRD. For Spider, we adopt the execution accuracy (EX) and test-suite accuracy (TS) as the evaluation metrics. As for BIRD, we consider EX along with the valid efficiency score (VES) to gauge its performance. Then, we fully fine-tune the parameters of CodeS using the training set for each benchmark and compare the supervised fine-tuned (SFT) CodeS models with several existing state-of-the-art text-to-SQL methods. The results are shown below:

On spider's development set:
| Method | Type | Spider Dev (EX) | Spider Dev (TS) | 
|-------|--------|--------|--------|
| T5-3B + PICARD [1] | Fine-tuning | 79.3% | 69.4% |
| C3 + ChatGPT [2] | Prompting | 81.8% | 71.4% |
| RESDSQL-3B + NatSQL [3] | Fine-tuning | 84.1% | 73.5% |
| DIN-SQL + GPT-4 [4] | Prompting | 82.8% | 74.2% |
| Graphix-T5-3B + PICARD [5] | Fine-tuning | 81.0% | 75.0% |
| Fine-tuned SQL-PaLM [6] | Fine-tuning | 82.8% | 78.2% |
| SFT CodeS-1B | Fine-tuning | 77.9% | 72.2% |
| SFT CodeS-3B | Fine-tuning | 83.4% | 78.1% |
| SFT CodeS-7B | Fine-tuning | **85.4%** | **80.3%** |
| SFT CodeS-15B | Fine-tuning | 84.9% | 79.4% |

On BIRD's development set:
| Method | Orale Know. | Type | Bird Dev (EX) | Bird Dev (VES) |
|-------|--------|--------|--------|--------|
| Fine-tuned T5-3B [7] |  | Fine-tuning | 10.37% | 13.62% |
| ChatGPT [7] |  | Prompting | 24.05% | 27.97% |
| ChatGPT + CoT [7] |  | Prompting | 25.88% | 32.33% |
| SFT CodeS-1B |   | Fine-tuning | 38.46% | 41.77% |
| SFT CodeS-3B |   | Fine-tuning | 43.42% | 44.55% |
| SFT CodeS-7B |   | Fine-tuning | 45.24% | 48.13% |
| SFT CodeS-15B |   | Fine-tuning | **47.91%** | **49.60%** |
| Fine-tuned T5-3B [7] | ✔ | Fine-tuning | 23.34% | 25.57% |
| ChatGPT [7] | ✔ | Prompting | 42.24% | - |
| ChatGPT + CoT [7] | ✔ | Prompting | 36.64% | 42.30% |
| GPT-4 [7] | ✔ | Prompting | 49.15% | - |
| DIN-SQL + GPT-4 [4] | ✔ | Prompting | 50.72% | 58.79% |
| SFT CodeS-1B | ✔ | Fine-tuning | 50.46% | 51.07% |
| SFT CodeS-3B | ✔ | Fine-tuning | 55.02% | 56.54% |
| SFT CodeS-7B | ✔ | Fine-tuning | 57.17% | 58.80% |
| SFT CodeS-15B | ✔ | Fine-tuning | **58.47%** | **59.87%** |

Our SFT CodeS models have achieved a remarkable level of performance on the challenging Spider and BIRD benchmarks. Additionally, we provide a thorough assessment of the robustness of our text-to-SQL models across various benchmarks, such as **Spider-DK, Spider-Syn, Spider-Realistic, and Dr.Spider**. For detailed insights into these findings, please consult the experimental section of our paper.

## Reproduce our results
Now, you can effortlessly replicate the results by utilizing the checkpoints and scripts we've provided.

### Step1: prepare environments
First, you should create the Anaconda environment and install the required modules:
```
conda create -n codes python=3.8.5
conda activate codes
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```
Then, you'll need to install SimCSE, a tool employed for retrieving relevant demonstrations in the few-shot scenario:
```
git clone https://github.com/lihaoyang-ruc/SimCSE.git
cd SimCSE
python setup.py install
cd ..
```
Lastly, make sure to download the necessary datasets (including Spider and Bird) [data.zip](https://drive.google.com/file/d/1-tfTMpc4gEtPqje_9jv-NU4csPQQz622/view?usp=sharing), the schema item classifier checkpoints [sic_ckpts.zip](https://drive.google.com/file/d/1V3F4ihTSPbV18g3lrg94VMH-kbWR_-lY/view?usp=sharing), and Spider's evaluation scripts [test_suite_sql_eval.zip](https://drive.google.com/file/d/1HIKBL7pP_hzWH1ryRNsjPO-N__UluOlK/view?usp=sharing). Once downloaded, simply unzip these files in the root folder.
```
unzip data.zip
unzip sic_ckpts.zip
unzip test_suite_sql_eval.zip
```

### Step2: run inference scripts
**Few-shot CodeS**

For your convenience, we offer a script `text2sql_few_shot.py` to facilitate the reproduction of the few-shot results achieved by CodeS:
```
python -u text2sql_few_shot.py --model_path [model path] --sic_path [sic path] --dataset_path [dataset path] --demonstration_set_path [demon set path] --num_of_demonstrations [num of demon] --load_in_4bit --load_in_8bit

arguments:
    [model path]     path (or huggingface name) of the LLM
    [sic path]       path of the schema item classifier
    [dataset path]   path of the evaluation set
    [demon set path] path of the demonstration pool (i.e., the training set)
    [num of demon]   number of demonstrations in the context
    --load_in_4bit   load LLM in 4bit quantization (less GPU memory but slower inference speed)
    --load_in_8bit   load LLM in 8bit quantization (less GPU memory but slower inference speed)
```

Here are the commands that will enable you to obtain the results for CodeS in the 3-shot setting:
```
# Spider
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --model_path {model_path} --sic_path ./sic_ckpts/sic_spider --dataset_path ./data/sft_eval_spider_text2sql.json --demonstration_set_path ./data/sft_train_spider_text2sql.json --num_of_demonstrations 3

# Bird
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --model_path {model_path} --sic_path ./sic_ckpts/sic_bird --dataset_path ./data/sft_eval_bird_text2sql.json --demonstration_set_path ./data/sft_train_bird_text2sql.json --num_of_demonstrations 3

# Bird (using orale knowledge)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --model_path {model_path} --sic_path ./sic_ckpts/sic_bird_with_evidence --dataset_path ./data/sft_eval_bird_with_evidence_text2sql.json --demonstration_set_path ./data/sft_train_bird_with_evidence_text2sql.json --num_of_demonstrations 3
```
You have the flexibility to select the `--model_path` argument from the following options:
| CodeS-1B | CodeS-3B | CodeS-7B | CodeS-15B|
| -------- | -------- | -------- | -------- |
| [seeklhy/codes-1b](https://huggingface.co/seeklhy/codes-1b) | [seeklhy/codes-3b](https://huggingface.co/seeklhy/codes-3b) | [seeklhy/codes-7b](https://huggingface.co/seeklhy/codes-7b) | [seeklhy/codes-15b](https://huggingface.co/seeklhy/codes-15b) |

For example, if you want to reproduce the results of 3-shot CodeS-15B on Spider, you can run:
```
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --model_path seeklhy/codes-15b --sic_path ./sic_ckpts/sic_spider --dataset_path ./data/sft_eval_spider_text2sql.json --demonstration_set_path ./data/sft_train_spider_text2sql.json --num_of_demonstrations 3
```

**SFT CodeS**

Additionally, we supply a script named `text2sql_zero_shot.py`, which facilitates the acquisition of results for the fine-tuned CodeS:
```
python -u text2sql_zero_shot.py --model_path [model path] --dataset_path [dataset path] --sic_path [sic path] --load_in_4bit --load_in_8bit

arguments:
    [model path]     path (or huggingface name) of the LLM
    [dataset path]   path of the evaluation set
    [sic path]       path of the schema item classifier
    --load_in_4bit   load LLM in 4bit quantization (less GPU memory but slower inference speed)
    --load_in_8bit   load LLM in 8bit quantization (less GPU memory but slower inference speed)
```

To obtain the results of the fine-tuned CodeS, simply execute the following commands:
```
# Spider
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --model_path {spider_model_path} --dataset_path ./data/sft_eval_spider_text2sql.json --sic_path ./sic_ckpts/sic_spider

# BIRD
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --model_path {bird_model_path} --dataset_path ./data/sft_eval_bird_text2sql.json --sic_path ./sic_ckpts/sic_bird

# BIRD (using orale knowledge)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --model_path {bird_with_evidence_model_path} --dataset_path ./data/sft_eval_bird_with_evidence_text2sql.json --sic_path ./sic_ckpts/sic_bird_with_evidence
```
You can choose the `--model_path` argument from the available options:
| {spider_model_path} | {bird_model_path} | {bird_with_evidence_model_path} |
| ------ | ------ | ----------- |
| [seeklhy/codes-1b-spider](https://huggingface.co/seeklhy/codes-1b-spider) | [seeklhy/codes-1b-bird](https://huggingface.co/seeklhy/codes-1b-bird) | [seeklhy/codes-1b-bird-with-evidence](https://huggingface.co/seeklhy/codes-1b-bird-with-evidence) |
| [seeklhy/codes-3b-spider](https://huggingface.co/seeklhy/codes-3b-spider) | [seeklhy/codes-3b-bird](https://huggingface.co/seeklhy/codes-3b-bird) | [seeklhy/codes-3b-bird-with-evidence](https://huggingface.co/seeklhy/codes-3b-bird-with-evidence) |
| [seeklhy/codes-7b-spider](https://huggingface.co/seeklhy/codes-7b-spider) | [seeklhy/codes-7b-bird](https://huggingface.co/seeklhy/codes-7b-bird) | [seeklhy/codes-7b-bird-with-evidence](https://huggingface.co/seeklhy/codes-7b-bird-with-evidence) |
| [seeklhy/codes-15b-spider](https://huggingface.co/seeklhy/codes-15b-spider) | [seeklhy/codes-15b-bird](https://huggingface.co/seeklhy/codes-15b-bird) | [seeklhy/codes-15b-bird-with-evidence](https://huggingface.co/seeklhy/codes-15b-bird-with-evidence) |

For example, if you want to obtain the results of SFT CodeS-7B on the BIRD benchmark (using orale knowledge), you can use the following command:
```
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --model_path seeklhy/codes-7b-bird-with-evidence --dataset_path ./data/sft_eval_bird_with_evidence_text2sql.json --sic_path ./sic_ckpts/sic_bird_with_evidence
```

## Fine-Tune CodeS
(Awaiting addition.)

## Use CodeS on your data
If you intend to employ CodeS with your privacy data, you can fine-tune CodeS using your training data and deploy it on your local machine.

*Important Note: We should emphasize that CodeS is a pre-trained LLM and has not aligned with humans via supervised fine-tuning and reinforcement learning from human feedback. Consequently, CodeS does not fall within the category of a "Chat" LLM.*

## License
The code and weights in this repository are open-sourced under the Apache-2.0 license.

## Contact
CodeS is a collaborative effort between Renmin University of China and AI-Finance. If you have any questions, we encourage you to either create Github issues or get in touch directly with Haoyang Li at lihaoyang.cs@ruc.edu.cn.

## References
[1] Scholak, T., Schucher, N., & Bahdanau, D. (2021). PICARD: Parsing incrementally for constrained auto-regressive decoding from language models. arXiv preprint arXiv:2109.05093.

[2] Dong, X., Zhang, C., Ge, Y., Mao, Y., Gao, Y., Lin, J., & Lou, D. (2023). C3: Zero-shot Text-to-SQL with ChatGPT. arXiv preprint arXiv:2307.07306.

[3] Li, H., Zhang, J., Li, C., & Chen, H. (2023, June). Resdsql: Decoupling schema linking and skeleton parsing for text-to-sql. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 11, pp. 13067-13075).

[4] Pourreza, M., & Rafiei, D. (2023). Din-sql: Decomposed in-context learning of text-to-sql with self-correction. arXiv preprint arXiv:2304.11015.

[5] Li, J., Hui, B., Cheng, R., Qin, B., Ma, C., Huo, N., ... & Li, Y. (2023). Graphix-t5: Mixing pre-trained transformers with graph-aware layers for text-to-sql parsing. arXiv preprint arXiv:2301.07507.

[6] Sun, R., Arik, S. O., Nakhost, H., Dai, H., Sinha, R., Yin, P., & Pfister, T. (2023). SQL-PaLM: Improved Large Language ModelAdaptation for Text-to-SQL. arXiv preprint arXiv:2306.00739.

[7] [Bird's official leaderboard](https://bird-bench.github.io)
