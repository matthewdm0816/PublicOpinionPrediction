#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=true

python train_emergency.py
# accelerate launch --num_processes=1 --config_file yuqing.yaml train_distributed.py