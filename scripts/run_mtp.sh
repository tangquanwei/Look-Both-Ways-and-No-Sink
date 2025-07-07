#!/bin/bash
# This script is used to run MNTP/MLM training


# improvement: ***   good        (speed 0.0%, memory 7.0%)
export TINY_FLASH_ATTN=1
# improvement: ***** huge        (speed 8.7%, memory 16.2%)
export TINY_FUSED_RMSNORM=1
# improvement: **    normal      (speed 0.0%, memory 4.6%)
export TINY_FUSED_CROSSENTROPY=1
# improvement: *     minor       (speed 1.9%, memory 0.5%)
export TINY_FUSED_ROTARY=1
# improvement: ****  significant (speed 1.6%, memory 16.2%)
export TINY_FUSED_SWIGLU=1

model=outputs/base/tinyllama-1.1B
abbr=$(echo "$model" | sed 's/\/snapshots\/.*//g')
abbr=$(echo "$abbr" | sed 's/outputs\///g; s/\//_/g')
gguf_file=none
task=mntp # mlm # 

output_dir="outputs/${task}/${abbr}"
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
lr=5e-4

python experiments/run_mtp.py \
    --model_name_or_path ${model} \
    --gguf_file ${gguf_file} \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --task ${task} \
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
    --max_train_samples 1000 \
    --preprocessing_num_workers 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.015 \
    --learning_rate ${lr} \
    --do_train True \
    --do_eval True \
    --max_seq_length 16 \
    --mask_token_type blank \
    --data_collator_type default \
    --mlm_probability 0.2 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to none \
    --overwrite_output_dir False \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --max_steps 10000 \
    --save_total_limit 3 \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \