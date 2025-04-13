#!/bin/bash

# Inference, and generate output json file
model_path="/mnt/llama-2-7b"
model="/mnt/llama-2-7b"
model_arch="llama"
partial_weight=0.2
capacity=1.0
alpha=7
budget=0.6
no_skewing=${1}
base_name=$(basename "${model}")
if [ -z $no_skewing ]; then
  weight_path="../setup/weights/${base_name}_${partial_weight}"
else 
  weight_path="../setup/weights-no-skew/${base_name}_${partial_weight}"
fi
skewing_path="../setup/skewing_matrix/${base_name}.pt"

python -u answer.py \
  --model-name ${model} \
  --model-type ${model_arch} \
  --partial_weight_ratio ${partial_weight} \
  --partial_weight_path ${weight_path} \
  --ours \
  --model-path ${model_path} \
  --skewing_matrix_path ${skewing_path} \
  --alpha ${alpha} \
  --capacity ${capacity} \
  --budget ${budget}
