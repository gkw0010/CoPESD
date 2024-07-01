#!/bin/bash



pretrained_type=consolidated
pretrained_path=accessory/output/llama_ens_light_13b_esd/epoch0
llama_config="$pretrained_path"/config.json
tokenizer_path="$pretrained_path"/tokenizer.model

llama_type=llama_ens_light

exp_name=eval/eval_llama_ens_light_13b_esd ####
save_name=eval_llama_ens_light_13b_esd
data_parallel=sdp
model_parallel=2
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"
MASTER_PORT=$((RANDOM % 101 + 20000))


torchrun --nproc_per_node 8 --master_port $((RANDOM % 101 + 20000)) generate_action.py \
--batch_size 8 \
--dataset coco   \
--llama_type "$llama_type" --llama_config $llama_config --tokenizer_path "$tokenizer_path" \
--pretrained_path "$pretrained_path" \
--savename "$save_name"     \
2>&1 | tee -a output/"$exp_name"/output.log 
echo "pretrained path: $pretrained_path"


