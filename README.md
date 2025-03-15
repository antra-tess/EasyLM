# EasyLM
Large language models (LLMs) made easy, EasyLM is a one stop solution for
pre-training, finetuning, evaluating and serving LLMs in JAX/Flax. EasyLM can
scale up LLM training to hundreds of TPU/GPU accelerators by leveraging
JAX's pjit functionality.


Building on top of Hugginface's [transformers](https://huggingface.co/docs/transformers/main/en/index)
and [datasets](https://huggingface.co/docs/datasets/index), this repo provides
an easy to use and easy to customize codebase for training large language models
without the complexity in many other frameworks.


EasyLM is built with JAX/Flax. By leveraging JAX's pjit utility, EasyLM is able
to train large models that don't fit on a single accelerator by sharding
the model weights and training data across multiple accelerators. Currently,
EasyLM supports multiple TPU/GPU training in a single host as well as multi-host
training on Google Cloud TPU Pods.

Currently, the following models are supported:
* [LLaMA](https://arxiv.org/abs/2302.13971)
* [LLaMA 2](https://arxiv.org/abs/2307.09288)
* [LLaMA 3](https://llama.meta.com/llama3/)

## Discord Server
We are running an unofficial Discord community (unaffiliated with Google) for discussion related to training LLMs in JAX. [Follow this link to join the Discord server](https://discord.gg/Rf4drG3Bhp). We have dedicated channels for several JAX based LLM frameworks, include EasyLM, [JaxSeq](https://github.com/Sea-Snell/JAXSeq), [Alpa](https://github.com/alpa-projects/alpa) and [Levanter](https://github.com/stanford-crfm/levanter).


## Models Trained with EasyLM
### OpenLLaMA
OpenLLaMA is our permissively licensed reproduction of LLaMA which can be used
for commercial purposes. Check out the [project main page here](https://github.com/openlm-research/open_llama).
The OpenLLaMA can serve as drop in replacement for the LLaMA weights in EasyLM.
Please refer to the [LLaMA documentation](docs/llama.md) for more details.


### Koala
Koala is our new chatbot fine-tuned on top of LLaMA. If you are interested in
our Koala chatbot, you can check out the [blogpost](https://bair.berkeley.edu/blog/2023/04/03/koala/)
and [documentation for running it locally](docs/koala.md).


## Installation
The installation method differs between GPU hosts and Cloud TPU hosts. The first
step is to pull from GitHub.

``` shell
git clone https://github.com/young-geng/EasyLM.git
cd EasyLM
export PYTHONPATH="${PWD}:$PYTHONPATH"
```

#### Installing on GPU Host
The GPU environment can be installed via [Anaconda](https://www.anaconda.com/products/distribution).

``` shell
conda env create -f scripts/gpu_environment.yml
conda activate EasyLM
```

#### Installing on Cloud TPU Host
The TPU host VM comes with Python and PIP pre-installed. Simply run the following
script to set up the TPU host.

``` shell
./scripts/tpu_vm_setup.sh
```


## [Documentations](docs/README.md)
The EasyLM documentations can be found in the [docs](docs/) directory.


## Reference
If you found EasyLM useful in your research or applications, please cite using the following BibTeX:
```
@software{geng2023easylm,
  author = {Geng, Xinyang},
  title = {EasyLM: A Simple And Scalable Training Framework for Large Language Models},
  month = March,
  year = 2023,
  url = {https://github.com/young-geng/EasyLM}
}
```



## Credits
* The LLaMA implementation is from [JAX_llama](https://github.com/Sea-Snell/JAX_llama)
* The JAX/Flax GPT-J and RoBERTa implementation are from [transformers](https://huggingface.co/docs/transformers/main/en/index)
* Most of the JAX utilities are from [mlxu](https://github.com/young-geng/mlxu)
* The codebase is heavily inspired by [JAXSeq](https://github.com/Sea-Snell/JAXSeq)

# Gemma-3-27B Fine-Tuning with Hugging Face

This repository contains scripts for fine-tuning the Google Gemma-3-27B model using Hugging Face's Transformers and PEFT libraries. The setup is optimized for 2x A100 80GB GPUs.

## Setup Overview

- **Model**: google/gemma-3-27b-pt (27B parameters)
- **Training Method**: LoRA fine-tuning with 4-bit quantization
- **Hardware**: 2x NVIDIA A100 80GB GPUs
- **Distributed Training**: DeepSpeed ZeRO-3 with parameter and optimizer offloading

## Requirements

Make sure you have:

1. 2x A100 80GB GPUs
2. CUDA 12.x installed
3. Python 3.10+
4. Hugging Face account with access to Gemma-3 models
5. WANDB account (optional but recommended for tracking)

## Installation

Install dependencies with the provided script:

```bash
bash install_requirements.sh
```

## Data Format

The training scripts expect a JSONL file with examples and a YAML template file. The default template format is:

```yaml
sequence:
  - no_loss: "{instruction}{input}\n"
  - no_loss: '<msg username="{author}">'
  - with_loss: "{output}"
  - with_loss: '</msg>\n'
```

Make sure your JSONL file has the corresponding fields (instruction, input, author, output).

## Training

To run the training:

```bash
bash run_gemma_sft.sh
```

This script:
1. Creates necessary directories
2. Configures DeepSpeed ZeRO-3 with optimal settings for 2x A100 GPUs
3. Launches distributed training with 4-bit quantization
4. Uses gradient checkpointing and other memory optimizations
5. Logs metrics to WANDB

## Key Configuration Settings

For 2x A100 80GB GPUs:
- Batch size: 1 per GPU
- Gradient accumulation steps: 16
- Effective batch size: 32 (1 × 2 GPUs × 16 accumulation steps)
- 4-bit quantization (NF4)
- Gradient checkpointing enabled
- LoRA rank: 32, alpha: 64

## Inference

To run inference with your fine-tuned model:

```bash
python inference.py --adapter_path /mnt/disk2/gemma_sft_output --load_in_4bit --interactive
```

## Additional Tools

- **trl_example.py**: Example script for TRL (RLHF) fine-tuning
- **merge_lora.py**: Utility to merge LoRA weights with the base model

## Memory Considerations

The configuration is optimized for 2x A100 80GB GPUs. Key memory optimizations:
- 4-bit quantization
- DeepSpeed ZeRO-3 with CPU offloading
- Gradient checkpointing
- Small per-device batch size with gradient accumulation

## Troubleshooting

If you encounter out-of-memory errors:
1. Reduce per_device_train_batch_size to 1 (already set)
2. Increase gradient_accumulation_steps (e.g., 16 to 24)
3. Enable more aggressive CPU offloading in DeepSpeed config
4. Reduce sequence length if possible
