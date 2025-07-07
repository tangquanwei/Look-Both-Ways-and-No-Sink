model=outputs/base/tinyllama-1.1B #JaaackXD/Llama-3-8B-Instruct-GGUF
abbr=$(echo "$model" | sed 's/\/snapshots\/.*//g')
abbr=$(echo "$abbr" | sed 's/outputs\///g; s/\//_/g')
peft_file=none
gguf_file=none # model-Q5_K_M.gguf

architecture="INPLACE"
mask_type="MASK0"
num_unsink_layers=0
num_bidir_layers=0
output_dir="outputs/supervised/${abbr}"

if [ "$architecture" != "NONE" ]; then
    output_dir="${output_dir}_${backward_type}"
fi

if [ "$num_unsink_layers" != "0" ]; then
    output_dir="${output_dir}_${mask_type,,}_${num_unsink_layers}"
fi

if [ "$num_bidir_layers" != "0" ]; then
    output_dir="${output_dir}_bidir_${num_bidir_layers}"
fi

# num_samples = 64K
#             = 64 * 1000
#             = 8  * 8000
# warmup_steps, save_steps, logging_steps

python experiments/run_sup.py \
    --model_name_or_path ${model} \
    --peft_model_name_or_path ${peft_file} \
    --gguf_file ${gguf_file} \
    --architecture ${architecture} \
    --mask_type ${mask_type} \
    --num_unsink_layers ${num_unsink_layers} \
    --num_bidir_layers ${num_bidir_layers} \
    --pooling_mode mean \
    --dataset_name E5 \
    --dataset_file_path data/echo-data \
    --remove_unused_columns False \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --warmup_steps 300 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --do_train True \
    --disable_tqdm False \
    --max_seq_length 512 \
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