# CodeS: Towards Building Open-source Language Models for Text-to-SQL

We release **CodeS**, a series of **Code** LLMs specifically trained for **S**QL generation. CodeS is **incrementally pre-trained** based on StarCoder using a large SQL-related corpus. The CodeS series encompasses four scales: [CodeS-1B](https://huggingface.co/seeklhy/codes-1b), [CodeS-3B](https://huggingface.co/seeklhy/codes-3b), [CodeS-7B](https://huggingface.co/seeklhy/codes-7b), and [CodeS-15B](https://huggingface.co/seeklhy/codes-15b). Our CodeS models have demonstrated outstanding performance on challenging text-to-SQL benchmarks, including **Spider and BIRD**. Furthermore, we conduct a comprehensive evaluation of CodeS's robustness across various benchmarks, encompassing **Spider-DK, Spider-Syn, Spider-Realistic, and Dr.Spider**. For in-depth insights into these results, please refer to the experimental section of our paper.

Utilizing CodeS, we have launched a text-to-SQL demo. You can access it at [RUCKBReasoning/text2sql-demo](https://github.com/RUCKBReasoning/text2sql-demo). Feel free to explore and follow the provided instructions to customize your own text-to-SQL demo!

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

### Run Fine-Tuning
(Awaiting addition.)

## License
The code and model weights are open-sourced under the Apache-2.0 license.

## Contact
If you have any questions, we encourage you to either create Github issues or get in touch directly with Haoyang Li at lihaoyang.cs@ruc.edu.cn.