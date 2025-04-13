#!/bin/bash

CWD=${PWD}
cd ../transformers/src/transformers/models

model="llama"
mv ${model}/modeling_${model}.py ${model}/modeling_${model}_orig.py

cd ${CWD}
# model_name="llama-2-7b"
# model_name="llama-3.1-8b-instruct"
model_name="llama-3.2-3b-instruct"
LLAMA_PATH="/mnt/llama-model"

# ========= InfiniGen ============
# generate skewing matrices for llama

python gen_llama_skewing_matrix.py \
    --model "${LLAMA_PATH}/${model_name}" \
    --output "./skewing_matrix" 

# generate partial weight matrices for prediction
PARTIAL_RATIO=0.2
# llama
python gen_partial_weight.py \
    --skewing_matrix_path "./skewing_matrix/${model_name}.pt" \
    --model "${LLAMA_PATH}/${model_name}" \
    --model_type "llama" \
    --partial_weight_ratio $PARTIAL_RATIO \
    --output "./weights"