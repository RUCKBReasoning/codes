# CodeS: Towards Building Open-source Language Models for Text-to-SQL

We release **CodeS**, a series of **Code** LLMs specifically trained for **S**QL generation. CodeS is **incrementally pre-trained** based on StarCoder using a large SQL-related corpus. The CodeS series encompasses four scales: [CodeS-1B](https://huggingface.co/seeklhy/codes-1b), [CodeS-3B](https://huggingface.co/seeklhy/codes-3b), [CodeS-7B](https://huggingface.co/seeklhy/codes-7b), and [CodeS-15B](https://huggingface.co/seeklhy/codes-15b). Our carefully collected pre-training corpus is also available at [here](https://drive.google.com/file/d/1UVkwQU9pYWU_-hhQIpgH8xWpLAqm1StX/view?usp=sharing). 

CodeS series have demonstrated outstanding performance on many challenging text-to-SQL benchmarks, including **Spider and BIRD**. Furthermore, we conduct a comprehensive evaluation of CodeS's robustness across various benchmarks, encompassing **Spider-DK, Spider-Syn, Spider-Realistic, and Dr.Spider**. For in-depth insights into these results, please refer to the experimental section of our paper.

Utilizing CodeS, we have launched a text-to-SQL demo. You can access it at [RUCKBReasoning/text2sql-demo](https://github.com/RUCKBReasoning/text2sql-demo). Feel free to explore and follow the provided instructions to customize your own text-to-SQL demo!

`Update (2024.4.19):` We are excited to announce the release of our newly developed schema filter, boasting 3 billion parameters and offering bilingual support for both Chinese and English. This tool is now available as an independent component and can be accessed at [text2sql-schema-filter](https://github.com/RUCKBReasoning/text2sql-schema-filter). If you're looking to enhance your text-to-SQL system with a schema filter, we encourage you to give it a try. 

## Reproduce our results
Reproducing our results is straightforward using the checkpoints and scripts we have supplied.

### Prepare Environments
Our experiments are conducted in the following environments:
- GPU: 8 * NVIDIA A800 with 80GB VRAM, CUDA version 11.8
- CPU: Intel(R) Xeon(R) Platinum 8358 CPU, accompanied by 1024GB of RAM
- Operating System: CentOS Linux 7
- Python Environment: Anaconda3, Python version 3.8.5

#### Step1: Install Java
```
apt-get update
apt-get install -y openjdk-11-jdk
```
If you already have a Java environment installed, you can skip this step.

#### Step2: Create Python Environments
Create a new Anaconda environment and install the required modules:
```
conda create -n codes python=3.8.5
conda activate codes
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
git clone https://github.com/lihaoyang-ruc/SimCSE.git
cd SimCSE
python setup.py install
cd ..
```

#### Step3: Download Datasets, Checkpoints, and Spider's Evaluation Scripts
Download the necessary datasets [data.zip](https://drive.google.com/file/d/189spLXUL3gF8k4sny5qiWMqW3wOzx5AD/view?usp=sharing), the schema item classifier checkpoints [sic_ckpts.zip](https://drive.google.com/file/d/1V3F4ihTSPbV18g3lrg94VMH-kbWR_-lY/view?usp=sharing), and the Spider's evaluation scripts [test_suite_sql_eval.zip](https://drive.google.com/file/d/1iNa1WgA9tN_OFna08nq_tHZdXx9Lz2vO/view?usp=sharing). Then, unzip them using the following commands:
```
unzip data.zip
unzip sic_ckpts.zip
unzip test_suite_sql_eval.zip
```

#### Step4: Pre-process data
You can skip this step as the pre-processed datasets are already included in the aforementioned `data.zip` file. However, if you wish to reproduce our data pre-processing procedure, you can execute the following two Python scripts:
```
# build BM25 index for each database
python -u build_contents_index.py
# pre-process dataset
python -u prepare_sft_datasets.py
```
Please note that this process may take a considerable amount of time (approximately 1-2 hours). Your patience is appreciated.

### Run Inference
We offer two inference scripts, namely `run_few_shot_evaluations.sh` and `run_sft_evaluations.sh`, to facilitate the reproduction of our few-shot in-context learning and SFT results as reported in the paper.

## Training on Text-to-SQL Benchmarks
For those looking to train CodeS on their own machines, we offer detailed scripts to guide you through the process.

The fine-tuning procedure is divided into two primary phases: first, fine-tuning the schema item classifiers (also referred to as schema filters in our study), and second, fine-tuning the CodeS models themselves.

### Stage1: Train Schema Item Classifier
To facilitate the training of schema item classifiers on the Spider, BIRD, and BIRD dataset augmented with external knowledge, we have included specific commands within the `train_sic.sh` script.

### Stage2: Train CodeS on Text-to-SQL benchmarks
Further, the `train_codes.sh` script provides commands for training CodeS models. This includes not only training on the Spider, BIRD, and BIRD dataset enhanced with external knowledge but also on the Bank_Financials, Aminer_Simplified, and the comprehensive all-merged dataset.

It’s important to mention that we utilize DeepSpeed Zero for data parallelism, offering GPU memory savings compared to traditional Distributed Data Parallel (DDP) approaches. Specifically, our implementation integrates DeepSpeed with [Hugging Face’s Accelerate](https://huggingface.co/docs/accelerate/usage_guides/deepspeed). To ensure a smooth training process, it’s necessary to configure Accelerate by executing `accelerate config` in your terminal, where you can set up your preferred configurations. Below are the options I selected:


```
$ accelerate config
In which compute environment are you running?
This machine

Which type of machine are you using?
multi-GPU

How many different machines will you use (use more than 1 for multi-node training)? [1]: 1

Do you wish to optimize your script with torch dynamo?[yes/NO]:NO

Do you want to use DeepSpeed? [yes/NO]: yes

Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO

What should be your DeepSpeed's ZeRO optimization stage?
2

Where to offload optimizer states?
none                                                         

Where to offload parameters?
none

How many gradient accumulation steps you're passing in your script? [1]: 4

Do you want to use gradient clipping? [yes/NO]: yes

What is the gradient clipping value? [1.0]: 1.0

Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: NO

How many GPU(s) should be used for distributed training? [1]:8

Do you wish to use FP16 or BF16 (mixed precision)?
bf16                                             
```

We’re equipped with 8 GPUs for our training processes, each capable of handling a batch size of 4. Additionally, we’ve configured the gradient accumulation steps to 4, culminating in a global batch size of $8 \times 4 \times 4=128$.

Should your GPUs be compatible with Flash-Attention (further details available at [flash-attention](https://github.com/Dao-AILab/flash-attention)), activating this feature can further reduce the GPU memory usage during training. To leverage Flash-Attention, we offer two approaches: **Modifying the source code within the transformers package** or **Installing the latest version of the transformers package**.

### Option 1: Modify the Source Code
To incorporate our modifications for GPTBigCode in the transformers package, you'll need to first back up the original file and then replace it with our modified version:
1. Back up the original `modeling_gpt_bigcode.py` file:
    ```
    mv your_python_env_path/site-packages/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py your_python_env_path/site-packages/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py-bk
    ```
2. Copy our modified version into the same directory:
    ```
    cp modeling_gpt_bigcode.py your_python_env_path/site-packages/transformers/models/gpt_bigcode
    ```
Afterwards, proceed to install the `flash-attn` package:
```
pip install flash-attn==2.0.0.post1
```

### Option 2: Upgrade the transformers Package
The latest version of the transformers package now includes Flash-Attention for GPTBigCode by default. Upgrading is straightforward:
1. Update the transformers package:
   ```
   pip install transformers==4.38.2
   ```
2. Install the `flash-attn` package:
   ```
   pip install flash-attn
   ```
This method simplifies the process by utilizing the in-built support for Flash-Attention in the recent transformers release. To activate Flash-Attention in the code, incorporate the parameter `attn_implementation = "flash_attention_2"` when invoking the `AutoModelForCausalLM.from_pretrained()` function. For additional information and guidance, please consult the following resource: [Combining Starcoder and Flash-Attention 2](https://huggingface.co/docs/transformers/model_doc/gpt_bigcode#combining-starcoder-and-flash-attention-2).

*However, during our experimentation, we observed that the most recent version of the transformers package can yield inference outcomes that marginally deviate from those obtained with the version utilized in this project.* Consequently, we suggest opting for the first method or establishing a new Anaconda environment specifically for installing the latest transformers package.

## Incremental Pre-training
To begin pre-training CodeS with our released corpus, please adhere to the instructions provided below:

Initially, ensure that you have sufficient computational resources available, as pre-training is resource-consuming and time-consuming.

Then, download our collected corpus from [pre-training-corpus](https://drive.google.com/file/d/1UVkwQU9pYWU_-hhQIpgH8xWpLAqm1StX/view?usp=sharing) and unzip it.

Next, execute the following script to tokenize the corpus:
```
python -u tokenize_pt_corpus.py
```

Lastly, the `pre_train.sh` file contains exmaple commands for launching the pre-training process. You simply need to adjust the `per_device_train_batch_size` and configure the Accelerate settings to suit your hardware environment. This step is essential to achieve a global batch size that includes exactly 4,194,304 tokens.

## License
The code and model weights are open-sourced under the Apache-2.0 license.

## Contact
If you have any questions, we encourage you to either create Github issues or get in touch directly with Haoyang Li at lihaoyang.cs@ruc.edu.cn.
