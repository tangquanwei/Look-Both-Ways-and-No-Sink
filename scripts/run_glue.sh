#!/bin/bash
# This script is used to run GLUE training, evaluation, and prediction.

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

# Model Abbr      | Parameters    | Name on Huggingface
# Roberta-large   | 354,319,369   | princeton-nlp/sup-simcse-roberta-large
# GPT2            | 139,219,986   | gpt2
# Tinyllama-1.1B  | 1,078,575,113 | tinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T(TinyLlama)
# Llama-3-8B      | 7,504,961,545 | meta-llama/Llama-3-8B

# Task Name | Description 
# cola      | Corpus of Linguistic Acceptablity (Gramma; Correct/Wrong)
# sst2      | Stanford Sentiment Treebank (Movie Review; Positive/Negative)
# mrpc      | Microsoft Research Paraphrase Corpus (News Sementic)
# qqp       | Quora Question Pairs (Question Semantic)
# stsb      | Semantic Textual Similarity Benchmark (Sentence Semantic; 1-5)
# mnli      | Multi-Genre NLI
# qnli      | Question Answering NLI (Question->Answer)
# rte       | Recognizing Textual Entailment
# wnli      | Winograd NLI (Coreference)

model=outputs/base/tinyllama-1.1B
abbr=$(echo "$model" | sed 's/\/snapshots\/.*//g')
abbr=$(echo "$abbr" | sed 's/outputs\///g; s/\//_/g')
gguf_file=none
task_name=sst2

output_dir="outputs/glue_${task_name}/${abbr}"
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
# GPT2           |   5e-4   |   5e-3    |      1e-3        |        1e-3        |
# Roberta-large  |   9e-5   |   5e-3    |      1e-3        |        1e-3        |
# Tinyllama-1.1B |   5e-3   |   5e-3    |      1e-3        |        1e-3        |
# -------------------------------------------------------------------------------

lr=5e-3 # Please Manually set the Optimal LR

python experiments/run_glue.py \
  --model_name_or_path ${model} \
  --gguf_file ${gguf_file} \
  --task_name ${task_name} \
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
  --max_seq_length 128 \
  --learning_rate ${lr} \
  --ignore_mismatched_sizes True \