model=outputs/base/tinyllama-1.1B
abbr=$(echo "$model" | sed 's/\/snapshots\/.*//g')
abbr=$(echo "$abbr" | sed 's/outputs\///g; s/\//_/g')
peft_file=none
gguf_file=none

architecture="INPLACE"
mask_type="MASK0"
num_unsink_layers=0
num_bidir_layers=0
unsink_layers="[]"
bidir_layers="[]"
output_dir="outputs/simcse/${abbr}"

if [ "$architecture" != "NONE" ]; then
    output_dir="${output_dir}_${backward_type}"
fi

if [ "$num_unsink_layers" != "0" ]; then
    output_dir="${output_dir}_${mask_type,,}_${num_unsink_layers}"
fi

if [ "$num_bidir_layers" != "0" ]; then
    output_dir="${output_dir}_bidir_${num_bidir_layers}"
fi

python experiments/run_simcse.py \
    --model_name_or_path ${model} \
    --peft_model_name_or_path ${peft_file} \
    --simcse_dropout 0.3 \
    --architecture ${architecture} \
    --mask_type ${mask_type} \
    --num_unsink_layers ${num_unsink_layers} \
    --num_bidir_layers ${num_bidir_layers} \
    --pooling_mode mean \
    --dataset_name Wiki1M \
    --dataset_file_path data/wiki1m_for_simcse.txt \
    --remove_unused_columns False \
    --learning_rate 3e-5 \
    --loss_scale 20 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --do_train True \
    --disable_tqdm False \
    --max_seq_length 128 \
    --overwrite_output_dir True \
    --output_dir ${output_dir} \
    --logging_steps 50 \
    --save_steps 200 \
    --save_only_model True \
    --max_steps 1000 \
    --lora_r 16 \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --seed 42