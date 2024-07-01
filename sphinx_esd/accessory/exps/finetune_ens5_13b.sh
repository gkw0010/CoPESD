#!/bin/bash


pretrained_path=data/SPHINX-Tiny-1k
pretrained_type=consolidated
llama_config="data/SPHINX-Tiny-1k/config.json"
tokenizer_path="data/SPHINX-Tiny-1k/tokenizer.model"
data_config=configs/data/train.yaml
llama_type=llama_ens5_light


data_parallel=sdp
model_parallel=2

exp_name=llama_ens5_light_13b_esd
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

torchrun --master_port=1112 --nproc_per_node=8 main_finetune.py \
--output_dir output/"$exp_name" --epochs 1 --warmup_epochs 0.01 \
--log_dir output_dir/"$exp_name" \
--batch_size 4 --accum_iter 4 --num_workers 4 \
--max_words 3072 \
--lr 2e-5 --min_lr 0 --clip_grad 8 --weight_decay 0 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type $llama_type --llama_config "$llama_config" --tokenizer_path "$tokenizer_path" \
--pretrained_path "$pretrained_path" --pretrained_type="$pretrained_type" \
--dialog    \
--data_config $data_config \
--image_transform only_norm \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"


