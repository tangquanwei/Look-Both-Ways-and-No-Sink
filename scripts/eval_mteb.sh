#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=false

model=outputs/base/tinyllama-1.1B
abbr=$(echo "$model" | sed 's/\/snapshots\/.*//g')
abbr=$(echo "$abbr" | sed 's/outputs\///g; s/\//_/g')
gguf_file=none
peft_file=none

pool_type=avg # last # 
prefix_type=instruction # query_or_passage #

architecture="INPLACE"
mask_type="MASK0"
num_unsink_layers=2
num_bidir_layers=15
unsink_layers="[]"
bidir_layers="[]"
model_init=true

output_dir="outputs/mteb/${abbr}"

if [ "$architecture" != "NONE" ]; then
    output_dir="${output_dir}_${architecture}"
fi

if [ "$num_unsink_layers" != "0" ]; then
    output_dir="${output_dir}_${mask_type,,}_${num_unsink_layers}"
fi

if [ "$num_bidir_layers" != "0" ]; then
    output_dir="${output_dir}_bidir_${num_bidir_layers}"
fi

output_dir="${output_dir}_${prefix_type}_${pool_type}"

python evaluations/mteb_except_retrieval_eval.py \
    --model_name_or_path ${model} \
    --tasks TASK_SUBSET \
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

python evaluations/mteb_beir_eval.py \
    --model_name_or_path ${model} \
    --tasks TASK_SUBSET \
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
