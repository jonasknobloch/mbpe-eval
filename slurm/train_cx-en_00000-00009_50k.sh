#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --mincpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=alpha
#SBATCH --gres=gpu:8
#SBATCH --time=120:00:00
#SBATCH --account=p_scads_nlp
#SBATCH --job-name=morph-gpt_gpt2_cx-en_00000-00009_50k

source /software/rapids/r23.10/Anaconda3/2023.07-2/etc/profile.d/conda.sh

conda activate /home/jknobloc/morph-gpt/env

export WANDB_PROJECT="morph-gpt_gpt2_cx-en_00000-00009"
export HF_HOME="/home/jknobloc/morph-gpt/.hf"

NOW=$(date "+%F_%H%M%S")

python run_clm.py \
    --model_type gpt2 \
    --tokenizer_file ./tokenizers/tokenizer_gpt2_cx-en_00000-00000_50k.json \
    --token hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \
    --dataset_name uonlp/CulturaX \
    --dataset_config_name en \
    --dataset_data_files en/en_part_00000.parquet,en/en_part_00001.parquet,en/en_part_00002.parquet,en/en_part_00003.parquet,en/en_part_00004.parquet,en/en_part_00005.parquet,en/en_part_00006.parquet,en/en_part_00007.parquet,en/en_part_00008.parquet,en/en_part_00009.parquet \
    --validation_split_percentage 5 \
    --auto_find_batch_size \
    --do_train \
    --do_eval \
    --save_strategy steps \
    --save_steps 10000 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --num_train_epochs 1.0 \
    --report_to wandb \
    --output_dir ./out/${NOW:-0000_000000}/gpt2_cx-en_00000-00009_50k
