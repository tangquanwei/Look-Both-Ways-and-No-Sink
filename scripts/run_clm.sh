#!/bin/bash
# This script is used to run CLM training, evaluation, and prediction using TinyLlama-1.1B model.

echo "Start running..."

# improvement: ***** huge
export TINY_FLASH_ATTN=1
# improvement: ****  significant
export TINY_FUSED_RMSNORM=1
# improvement: *     minor
export TINY_FUSED_CROSSENTROPY=1
# improvement: *     minor
export TINY_FUSED_ROTARY=1
# improvement: **    a little bit
export TINY_FUSED_SWIGLU=1

model=outputs/base/tinyllama-1.1B
abbr=$(echo "$model" | sed 's/\/snapshots\/.*//g')
abbr=$(echo "$abbr" | sed 's/outputs\///g; s/\//_/g')
gguf_file=None

output_dir="outputs/clm/${abbr}"
architecture="INPLACE"
mask_type="MASK0"
num_unsink_layers=0
num_bidir_layers=0
unsink_layers="[]"
bidir_layers="[]"
freeze_type=backbone
unfreeze_layers=0
model_init=true
num_classifier_layers=1

masked_probability=0.15
replaced_probability=0.02

lr=5e-3
accelerate launch experiments/run_clm.py \
    --model_name_or_path ${model} \
    --gguf_file ${gguf_file} \
    --dataset_name JeanKaddour/minipile \
    --cache_dir cache_dir \
    --architecture ${architecture} \
    --mask_type ${mask_type} \
    --num_unsink_layers ${num_unsink_layers} \
    --num_bidir_layers ${num_bidir_layers} \
    --unsink_layers "${unsink_layers}" \
    --bidir_layers "${bidir_layers}" \
    --freeze_type ${freeze_type} \
    --num_unfreeze_layers ${unfreeze_layers} \
    --model_init ${model_init} \
    --num_classifier_layers ${num_classifier_layers} \
    --output_dir ${output_dir} \
    --preprocessing_num_workers 32 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --auto_find_batch_size False \
    --gradient_accumulation_steps 1 \
    --block_size 2048 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.015 \
    --learning_rate ${lr} \
    --weight_decay 1e-1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 1 \
    --save_total_limit 1 \
    --save_strategy steps \
    --save_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --logging_steps 50 \
    --max_steps 4000 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to none \
    --overwrite_output_dir True \
    --masked_probability ${masked_probability} \
    --replaced_probability ${replaced_probability} \
    --mask_token_type blank \
