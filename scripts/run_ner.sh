#!/bin/bash
# This script is used to run NER, POS, CHUNKING training, evaluation, and prediction.


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


# WARNING: It is known that using bfloat16 with fused kernels can lead to NaNs in the loss.
#          Further investigation is needed to understand the root cause of this issue.

#  Model Abbr        |   Params  | Name on Huggingface
#  Roberta-large     |   354M    | princeton-nlp/sup-simcse-roberta-large
#  GPT2              |   139M    | gpt2
#  Llama-135M        |   135M    | amd/AMD-Llama-135m 
#  Llama-160M        |   160M    | JackFram/llama-160m 
#  Llama-1.1B        |   1.1B    | tinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T(TinyLlama)
#  Llama-3.2-3B      |   3.0B    | alpindale/Llama-3.2-3B 
#  Llama-3-8B        |   7.5B    | meta-llama/Llama-3-8B

# S-Llama-1.3B       |   1.3B    | princeton-nlp/Sheared-LLaMA-1.3B
# Mistral-1.1B       |   1.1B    | optimum/mistral-1.1b-testing
# Phi-3.5-mini-Inst  |   3.7B    | microsoft/Phi-3.5-mini-instruct
# Olmo-1B            |   1.2B    | allenai/OLMo-1B
# Qwen2.5-0.5B       |   500M    | Qwen/Qwen2.5-0.5B
# Qwen2.5-0.5B-Inst  |   500M    | Qwen/Qwen2.5-0.5B-Instruct
# Qwen2.5-1.5B-Inst  |   1.5B    | Qwen/Qwen2.5-1.5B-Instruct

# Dataset   | Task
# Conll2003 | ner, pos, chunk
# Few_nerd  | ner, coarse_ner

model=outputs/base/tinyllama-1.1B
abbr=$(echo "$model" | sed 's/\/snapshots\/.*//g')
abbr=$(echo "$abbr" | sed 's/outputs\///g; s/\//_/g')
gguf_file=none # model-Q5_K_M.gguf
dataset="data/conll2003" # "data/few_nerd" # 
task=ner

output_dir="outputs/${dataset##*/}_${task}/${abbr}"
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


# --------------------------------Optimal LR ------------------------------------
# Model\Type     | TrainAll | FreezeAll | UnfreezeOneLayer | TwoLayerClassifier |
# Roberta-large  |   1e-4   |   5e-3    |      1e-3        |        1e-3        |
# GPT2           |   5e-4   |   5e-3    |      1e-3        |        1e-3        |
# Tinyllama-1.1B |   9e-5   |   5e-3    |      1e-3        |        1e-3        |
# -------------------------------------------------------------------------------

lr=5e-3 # Please Manually set the Optimal LR

# ner
python experiments/run_ner.py \
  --model_name_or_path ${model} \
  --gguf_file ${gguf_file} \
  --dataset_name ${dataset} \
  --dataset_config_name default \
  --task_name ${task} \
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
  --output_dir "${output_dir}" \
  --do_train \
  --do_eval \
  --bf16 \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --evaluation_strategy epoch \
  --report_to none \
  --overwrite_output_dir True \
  --save_strategy no \
  --save_total_limit 1 \
  --max_seq_length 32 \
  --learning_rate ${lr} \
  --ignore_non_entity False \
  --label_all_tokens True \
  --ignore_mismatched_sizes True \
  --use_instructions False \
  --trust_remote_code False \