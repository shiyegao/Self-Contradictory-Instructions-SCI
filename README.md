# Dissecting Dissonance: Benchmarking Large Multimodal Models Against Self-Contradictory Instructions

## Introduction
Official github repo of the ECCV24 paper, "[Dissecting Dissonance: Benchmarking Large Multimodal Models Against Self-Contradictory Instructions](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07483.pdf)".
- Website: https://selfcontradiction.github.io
- Dataset: https://huggingface.co/datasets/sci-benchmark/self-contradictory
    - In the paper, “SCI-Core (1%), SCI-Base (10%), and SCI-All (100%)” denote the small, medium, and full splits of the Hugging Face dataset, respectively.

This repo provides the code for generating Self-Contradictory-Instruction(**SCI**) dataset.

- `LL`: This section of code provides the generation scripts for Language-Language Conflicts
- `VL`: This section of code provides the generation scripts for Vision-Language Conflicts

## File Structure

The basic file structure is shown as follows:

```
SCI
├── LL
│   ├── *.yaml
│   ├── generate.py
│   ├── information_conflict.py
│   ├── instruction_conflict.py
│   ├── seeds.json
│   └── utils.py
├── README.md
└── VL
    ├── *.yaml
    ├── generate.py
    ├── imageNet_class.py
    ├── tools.py
    └── visionlanguage.py
```

## Installation

```shell
conda create -n SCI python=3.9 
conda activate SCI
conda install pyyaml tqdm openai numpy matplotlib
```

## Usage

Two scripts are available for the generation of SCI:

1. `./LL/generate.py`  generates Language-Language(**L-L**) conflicts.
2. `./VL/generate.py`  generates Vision-Language(**V-L**) conflicts.

### L-L conflict generation

`./LL/generate.py`  generates only one type of subtask at a time. Details about the generation process can be set with the following optional arguments:

`--task` must be 1, 2, 3 or 4, for generating OCR Conflict, Attribute Conflict, Exclusion Conflict and Forbidden Conflict.

`--total_conflicts`  sets the total number of conflicts required.

`--config` is the configure file for openai API.

`--dataset_path` is the target path to save the generated conflicts.

Here is an example that generates 2500 conflicts for Rule Conflict, and saves them in `../SCI/Rule`

```shell
python generate.py --task 1 --total_conflicts 2500 --config openai.yaml --dataset_path ../SCI/Rule
```

### V-L conflict generation

`./VL/generate.py`  generates only one type of subtask at a time. Details about the generation process can be set with the following optional arguments:

`--task` must be 1, 2, 3 or 4, for generating OCR Conflict, Figure Conflict, Geometric Conflict and Semantic Conflict.

`--total_num`  sets the total number of conflicts required, only valid for task1 and task3.

`--config` is the configure file for openai API.

`--target_dir` is the target path to save the generated conflicts.

Here is an example that generates 2000 Geometric Conflicts, and saves them in `../SCI/OCR`

```shell
python generate.py --task 3 --total_num 2000 --config openai.yaml --dataset_path ../SCI/OCR
```

### OPENAI API

To use scripts above, a configure file for openai API is required.

Specifically, a `.yaml` file with the following information is required.

Here is an example of `openai.yaml`.

```yaml
OPENAI_KEY: your_openai_key
MODEL_NAME: 'your desired model'
OPENAI_API_BASE: 'your api base'

MAX_RETRY : 10
```


## LMM Evaluation

Many LMMs are tested on SCI in our paper. You can access them through APIs or local deployment.

[GPT4-V/GPT-4、ChatGPT](https://chat.openai.com/)

[ChatGLM/GLM4](https://chatglm.cn/)

[Gemini](https://gemini.google.com/)

[SPHINX-v2](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX)

[LLaVA-1.5](https://github.com/haotian-liu/LLaVA)

[LLaMA-Adapter v2](https://github.com/OpenGVLab/LLaMA-Adapter)

[BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)

## Citation

If our code or models help your work, please cite our paper.
```
@inproceedings{gao2024dissecting,
  title={Dissecting dissonance: Benchmarking large multimodal models against self-contradictory instructions},
  author={Gao, Jin and Gan, Lei and Li, Yuankai and Ye, Yixin and Wang, Dequan},
  booktitle={European Conference on Computer Vision},
  pages={404--420},
  year={2024},
  organization={Springer}
}
```
