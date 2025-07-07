#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=false

# BIOMEDICINE
# Tasks: PubMedQA, ChemProt, RCT
# Model: outputs/base/Bio-Medical-Llama-3-8B

# FINANCE
# Tasks: ConvFinQA, FPB
# Model: outputs/base/finma-7b-nlp

# LAW
# Tasks: SCOTUS, ToS
# Model: outputs/base/llama3-8b-Lawyer

model=outputs/base/gte-Qwen2-7B-instruct # outputs/base/NV-Embed-v2 #outputs/base/llama3-8b-Lawyer
abbr=$(echo "$model" | sed 's/\/snapshots\/.*//g')
abbr=$(echo "$abbr" | sed 's/outputs\///g; s/\//_/g')
gguf_file=none
peft_file=none
tasks=BIOMEDICINE_TASK_LIST

pool_type=avg # last # 
prefix_type=instruction # query_or_passage #

architecture="INPLACE"
mask_type="MASK0"
num_unsink_layers=0
num_bidir_layers=0
unsink_layers="[]"
bidir_layers="[]"
model_init=true

output_dir="outputs/domain/${abbr}_${prefix_type}_${pool_type}"

if [ "$architecture" != "NONE" ]; then
    output_dir="${output_dir}_${architecture}"
fi

if [ "$num_unsink_layers" != "0" ]; then
    output_dir="${output_dir}_${mask_type,,}_${num_unsink_layers}"
fi

if [ "$num_bidir_layers" != "0" ]; then
    output_dir="${output_dir}_bidir_${num_bidir_layers}"
fi

python evaluations/domain_specific_eval.py \
    --model_name_or_path ${model} \
    --tasks ${tasks} \
    --prefix_type ${prefix_type} \
    --pool_type ${pool_type} \
    --output_dir ${output_dir} \
    --architecture ${architecture} \
    --mask_type ${mask_type} \
    --num_unsink_layers ${num_unsink_layers} \
    --num_bidir_layers ${num_bidir_layers} \
    --unsink_layers "${unsink_layers}" \
    --bidir_layers "${bidir_layers}" \
    --model_init ${model_init} \
    --peft_file ${peft_file} \
    --gguf_file ${gguf_file}