#!/bin/bash
export HF_TOKEN="hf_XXXX"
export CUDA_VISIBLE_DEVICES=0

methods=("FullKV" "PyramidKV" "SnapKV" "H2O" "StreamingLLM") # Support FullKV, PyramidKV, SnapKV, H2O, StreamingLLM
methods=("PyramidKV")
max_capacity_prompts=128 # 128,2048 in paper
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "eager"
model_path="meta-llama/Meta-Llama-3-8B-Instruct"
save_dir="results_longbench" # path to result save_dir

for method in "${methods[@]}"
do
    python3 run_longbench.py \
        --method ${method} \
        --model_path ${model_path} \
        --max_capacity_prompts ${max_capacity_prompts} \
        --attn_implementation ${attn_implementation} \
        --save_dir ${save_dir} \
        --use_cache True
done