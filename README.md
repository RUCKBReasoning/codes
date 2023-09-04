# CodeS: An open pre-trained large language model (LLM) for SQL generation

We release **CodeS**, a series of **Code** LLMs specifically optimized for **S**QL generation. CodeS is **incrementally pre-trained** based on StarCoder using a large SQL-related corpus. 

The CodeS series encompasses four distinct scales: [CodeS-1B](https://huggingface.co/seeklhy/codes-1b), [CodeS-3B](https://huggingface.co/seeklhy/codes-3b), [CodeS-7B](https://huggingface.co/seeklhy/codes-7b), and [CodeS-15B](https://huggingface.co/seeklhy/codes-15b). CodeS-1B, 3B, and 7B are based on StarCoderBase-1B, 3B, and 7B and support the max length of 8,192. Meanwhile, CodeS-15B, derived from StarCoder-15B, accommodates sequences of up to 6,144 tokens due to computational constraints. The corpus is collected from different sources such as [starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata), [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k), [StaQC](https://huggingface.co/datasets/koutch/staqc), and more.

The original purpose of developing CodeS is to validate whether it is possible to strengthen the base model's abilities in a single domain (such as SQL generation) using a minimal amount of domain-specific corpus (which, of course, is comparatively small in quantity when compared to the data required for pre-training from scratch). This is particularly friendly for private deployments since we do not need to worry about sensitive data being fed into closed-source LLMs.

In addition, we also introduce a new few-shot in-context learning Text-to-SQL framework. Within this framework, we employ a demonstration retriever that is sensitive to question patterns, enhancing its ability to dynamically retrieve helpful samples from the training set. Additionally, we incorporate a schema item classifier to remain relevant schema (i.e., tables and columns) according to the question. This strategic approach could facilitate the avoidance of unwieldy input sequences.

*A research paper is upcoming in which you can find more detailed information about CodeS!* ✨

## Evaluation
We evaluate CodeS on two challenging Text-to-SQL benchmarks: Spider and Bird. For Spider, we adopt the execution accuracy (EX) and test-suite accuracy (TS) as the evaluation metrics. As for Bird, we are considering EX along with the valid efficiency score (VES) to gauge its performance.

**Question 1: Does incremental pre-training significantly enhance StarCoder's SQL generation capability?**

To answer this question, we evaluate StarCoder and CodeS in the setting of 4-shot in-context learning. Let's delve into the following numerical results:

On Spider's development set:
| LLM | Spider Dev (EX) | Spider Dev (TS) |
|-------|--------|--------|
| StarCoderBase-1B | 57.5% | 49.3% | 
| CodeS-1B | 66.3% | 57.4% | 
| StarCoderBase-3B | 69.1% | 61.3% |
| CodeS-3B | 75.7% | 68.5% | 
| StarCoderBase-7B | 73.0% | 65.3% | 
| CodeS-7B | 78.8% | 72.1% | 
| StarCoderBase-15B | 77.1% | 69.5% | 
| StarCoder-15B | 77.7% | 68.6% | 
| CodeS-15B | **80.8%** | **72.4%** | 

On Bird's development set:
| LLM | Orale Know. | Bird Dev (EX) | Bird Dev (VES) |
|-------|--------|--------|---------|
| StarCoderBase-1B |  | 21.32% | 22.73% |
| CodeS-1B |  | 27.12% | 31.09% |
| StarCoderBase-3B |  | 26.86% | 29.82% |
| CodeS-3B |  | 31.10% | 35.86% |
| StarCoderBase-7B |  | 32.01% | 34.91% |
| CodeS-7B |  | 34.81% | 36.75% |
| StarCoderBase-15B |  | 35.33% | 39.58% |
| StarCoder-15B |    | 35.27% | 40.25% |
| CodeS-15B |    | **38.14%** | **40.80%** |
| StarCoderBase-1B | ✔ | 23.92% | 28.15% |
| CodeS-1B | ✔ | 30.57% | 37.18% |
| StarCoderBase-3B | ✔ | 32.92% | 36.35% |
| CodeS-3B | ✔ | 40.48% | 43.37% |
| StarCoderBase-7B | ✔ | 40.09% | 42.68% |
| CodeS-7B | ✔ | 43.29% | 47.27% |
| StarCoderBase-15B | ✔| 42.89% | 47.28% |
| StarCoder-15B |   ✔  | 42.83% | 47.44% |
| CodeS-15B |   ✔  | **45.57%** | **49.03%** |

The few-shot results underscore that CodeS outperforms StarCoder on both the Spider and Bird benchmarks in the few-shot setting, which demonstrates the effectiveness of the incrementally pre-training.

**Question 2: Can CodeS be comparable with previous state-of-the-art Text-to-SQL methods?**

To answer this question, we fully fine-tune the parameters of CodeS using the training sets. We then compare the fine-tuned CodeS with the existing state-of-the-art Text-to-SQL methods. The results are shown below:

On spider's development set:
| Method | Type | Spider Dev (EX) | Spider Dev (TS) | 
|-------|--------|--------|--------|
| T5-3B + PICARD [1] | Fine-tuning | 79.3% | 69.4% |
| C3 + ChatGPT [2] | Prompting | 81.8% | 71.4% |
| RESDSQL-3B + NatSQL [3] | Fine-tuning | 84.1% | 73.5% |
| DIN-SQL + GPT-4 [4] | Prompting | 82.8% | 74.2% |
| Graphix-T5-3B + PICARD [5] | Fine-tuning | 81.0% | 75.0% |
| Few-shot SQL-PaLM [6] | Prompting | 82.7% | 77.3% |
| Fine-tuned SQL-PaLM [6] | Fine-tuning | 82.8% | 78.2% |
| Fine-tuned CodeS-1B | Fine-tuning | 79.7% | 73.4% |
| Fine-tuned CodeS-3B | Fine-tuning | 82.6% | 77.4% |
| Fine-tuned CodeS-7B | Fine-tuning | **85.3%** | **79.8%** |
| Fine-tuned CodeS-15B | Fine-tuning | 84.5% | 79.5% |

On bird's development set (EX/VES):
| Method | Orale Know. | Type | Bird Dev (EX) | Bird Dev (VES) |
|-------|--------|--------|--------|--------|
| Fine-tuned T5-3B [7] |  | Fine-tuning | 10.37% | 13.62% |
| ChatGPT [7] |  | Prompting | 24.05% | 27.97% |
| ChatGPT + CoT [7] |  | Prompting | 25.88% | 32.33% |
| Fine-tuned CodeS-1B |   | Fine-tuning | 37.09% | 41.56% |
| Fine-tuned CodeS-3B |   | Fine-tuning | 40.87% | 46.01% |
| Fine-tuned CodeS-7B |   | Fine-tuning | 41.85% | 48.73% |
| Fine-tuned CodeS-15B |   | Fine-tuning | **44.13%** | **50.31%** |
| Fine-tuned T5-3B [7] | ✔ | Fine-tuning | 23.34% | 25.57% |
| ChatGPT [7] | ✔ | Prompting | 37.22% | 43.81% |
| ChatGPT + CoT [7] | ✔ | Prompting | 36.64% | 42.30% |
| GPT-4 [7] | ✔ | Prompting | 46.35% | 49.77% |
| DIN-SQL + GPT-4 [4] | ✔ | Prompting | 50.72% | **58.79%** |
| Fine-tuned CodeS-1B | ✔ | Fine-tuning | 49.93% | 53.93% |
| Fine-tuned CodeS-3B | ✔ | Fine-tuning | 52.67% | 56.03% |
| Fine-tuned CodeS-7B | ✔ | Fine-tuning | 53.78% | 56.99% |
| Fine-tuned CodeS-15B | ✔ | Fine-tuning | **54.69%** | 58.30% |

Evidently, the fine-tuned CodeS attains a groundbreaking level of performance on both the challenging Spider and Bird benchmarks.

*(The performance of baselines are derived from their papers or the official leaderboards.)*

## Reproduce our results
Now, you can effortlessly replicate the results by utilizing the checkpoints and scripts we've released.

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
Lastly, make sure to download the necessary datasets (including Spider and Bird) [data.zip](https://drive.google.com/file/d/1-tfTMpc4gEtPqje_9jv-NU4csPQQz622/view?usp=sharing), as well as the schema item classifier checkpoints [sic_ckpts.zip](https://drive.google.com/file/d/1V3F4ihTSPbV18g3lrg94VMH-kbWR_-lY/view?usp=sharing), along with Spider's evaluation scripts [test_suite_sql_eval.zip](https://drive.google.com/file/d/1HIKBL7pP_hzWH1ryRNsjPO-N__UluOlK/view?usp=sharing). Once downloaded, simply unzip these files in the root folder.
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

Here are the commands that will enable you to obtain the results for CodeS in the 4-shot setting:
```
# Spider
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --model_path {model_path} --sic_path ./sic_ckpts/sic_spider --dataset_path ./data/sft_eval_spider_text2sql.json --demonstration_set_path ./data/sft_train_spider_text2sql.json --num_of_demonstrations 4

# Bird
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --model_path {model_path} --sic_path ./sic_ckpts/sic_bird --dataset_path ./data/sft_eval_bird_text2sql.json --demonstration_set_path ./data/sft_train_bird_text2sql.json --num_of_demonstrations 4

# Bird (using orale knowledge)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --model_path {model_path} --sic_path ./sic_ckpts/sic_bird_with_evidence --dataset_path ./data/sft_eval_bird_with_evidence_text2sql.json --demonstration_set_path ./data/sft_train_bird_with_evidence_text2sql.json --num_of_demonstrations 4
```
You have the flexibility to select the `--model_path` argument from the following options:
| CodeS-1B | CodeS-3B | CodeS-7B | CodeS-15B|
| -------- | -------- | -------- | -------- |
| [seeklhy/codes-1b](https://huggingface.co/seeklhy/codes-1b) | [seeklhy/codes-3b](https://huggingface.co/seeklhy/codes-3b) | [seeklhy/codes-7b](https://huggingface.co/seeklhy/codes-7b) | [seeklhy/codes-15b](https://huggingface.co/seeklhy/codes-15b) |

For example, if you want to reproduce the results of 4-shot CodeS-15B on Spider, you can run:
```
CUDA_VISIBLE_DEVICES=0 python -u text2sql_few_shot.py --model_path seeklhy/codes-15b --sic_path ./sic_ckpts/sic_spider --dataset_path ./data/sft_eval_spider_text2sql.json --demonstration_set_path ./data/sft_train_spider_text2sql.json --num_of_demonstrations 4
```

**Fine-tuned CodeS**

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

# Bird
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --model_path {bird_model_path} --dataset_path ./data/sft_eval_bird_text2sql.json --sic_path ./sic_ckpts/sic_bird

# Bird (using orale knowledge)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --model_path {bird_with_evidence_model_path} --dataset_path ./data/sft_eval_bird_with_evidence_text2sql.json --sic_path ./sic_ckpts/sic_bird_with_evidence
```
You can choose the `--model_path` argument from the available options:
| {spider_model_path} | {bird_model_path} | {bird_with_evidence_model_path} |
| ------ | ------ | ----------- |
| [seeklhy/codes-1b-spider](https://huggingface.co/seeklhy/codes-1b-spider) | [seeklhy/codes-1b-bird](https://huggingface.co/seeklhy/codes-1b-bird) | [seeklhy/codes-1b-bird-with-evidence](https://huggingface.co/seeklhy/codes-1b-bird-with-evidence) |
| [seeklhy/codes-3b-spider](https://huggingface.co/seeklhy/codes-3b-spider) | [seeklhy/codes-3b-bird](https://huggingface.co/seeklhy/codes-3b-bird) | [seeklhy/codes-3b-bird-with-evidence](https://huggingface.co/seeklhy/codes-3b-bird-with-evidence) |
| [seeklhy/codes-7b-spider](https://huggingface.co/seeklhy/codes-7b-spider) | [seeklhy/codes-7b-bird](https://huggingface.co/seeklhy/codes-7b-bird) | [seeklhy/codes-7b-bird-with-evidence](https://huggingface.co/seeklhy/codes-7b-bird-with-evidence) |
| [seeklhy/codes-15b-spider](https://huggingface.co/seeklhy/codes-15b-spider) | [seeklhy/codes-15b-bird](https://huggingface.co/seeklhy/codes-15b-bird) | [seeklhy/codes-15b-bird-with-evidence](https://huggingface.co/seeklhy/codes-15b-bird-with-evidence) |

For instance, if you aim to obtain the results of fine-tuned CodeS-7B on the Bird benchmark (using orale knowledge), you can use the following command:
```
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py --model_path seeklhy/codes-7b-bird-with-evidence --dataset_path ./data/sft_eval_bird_with_evidence_text2sql.json --sic_path ./sic_ckpts/sic_bird_with_evidence
```

## Use CodeS on your data
If your intention is to employ CodeS with your privacy data, the choice of strategy should align with the volume of training data at your disposal:

- Few-shot in-context learning: In cases where you only have a limited number of training samples, opting for the few-shot in-context learning approach is advisable.

- Fully fine-tuning: On the other hand, if your training data are substantial, embracing a fully fine-tuning method for CodeS could yield enhanced performance outcomes. This strategy allows for a more comprehensive adaptation of CodeS's parameters to the available data.

*Important Note: We should emphasize that CodeS is a pre-trained LLM and has not aligned with humans via supervised fine-tuning and reinforcement learning from human feedback. Consequently, CodeS does not fall within the category of a "Chat" LLM.*

## Future work
1. Submit the fine-tuned CodeS to the leaderboards for Spider and Bird benchmarks.⚪
2. Evaluate the few-shot performance of closed-source LLMs like Text-davinci-003 and ChatGPT.⚪
3. Make available the training code, data pre-processing code, and the data employed for the incremental pre-training process.⚪

## License
The code and weights in this repository are open-sourced under the Apache-2.0 license.

## Contact
CodeS is a collaborative effort between Renmin University of China and ai-finance (金科览智). If you have any questions, we encourage you to either create Github issues or get in touch directly with Haoyang Li at lihaoyang.cs@ruc.edu.cn.

## References
[1] Scholak, T., Schucher, N., & Bahdanau, D. (2021). PICARD: Parsing incrementally for constrained auto-regressive decoding from language models. arXiv preprint arXiv:2109.05093.

[2] Dong, X., Zhang, C., Ge, Y., Mao, Y., Gao, Y., Lin, J., & Lou, D. (2023). C3: Zero-shot Text-to-SQL with ChatGPT. arXiv preprint arXiv:2307.07306.

[3] Li, H., Zhang, J., Li, C., & Chen, H. (2023, June). Resdsql: Decoupling schema linking and skeleton parsing for text-to-sql. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 11, pp. 13067-13075).

[4] Pourreza, M., & Rafiei, D. (2023). Din-sql: Decomposed in-context learning of text-to-sql with self-correction. arXiv preprint arXiv:2304.11015.

[5] Li, J., Hui, B., Cheng, R., Qin, B., Ma, C., Huo, N., ... & Li, Y. (2023). Graphix-t5: Mixing pre-trained transformers with graph-aware layers for text-to-sql parsing. arXiv preprint arXiv:2301.07507.

[6] Sun, R., Arik, S. O., Nakhost, H., Dai, H., Sinha, R., Yin, P., & Pfister, T. (2023). SQL-PaLM: Improved Large Language ModelAdaptation for Text-to-SQL. arXiv preprint arXiv:2306.00739.

[7] [Bird's official leaderboard](https://bird-bench.github.io)