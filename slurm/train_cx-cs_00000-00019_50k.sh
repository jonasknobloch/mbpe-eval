#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --mincpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=alpha
#SBATCH --gres=gpu:8
#SBATCH --time=120:00:00
#SBATCH --account=p_scads_nlp
#SBATCH --job-name=morph-gpt_gpt2_cx-cs_00000-00019_50k

source /software/rapids/r23.10/Anaconda3/2023.07-2/etc/profile.d/conda.sh

conda activate /home/jknobloc/morph-gpt/env

export WANDB_PROJECT="morph-gpt_gpt2_cx-cs_00000-00019"
export HF_HOME="/home/jknobloc/morph-gpt/.hf"

NOW=$(date "+%F_%H%M%S")

python run_clm.py \
    --model_type gpt2 \
    --tokenizer_file ./tokenizers/tokenizer_gpt2_cx-cs_00000-00001_50k.json \
    --token hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \
    --dataset_name uonlp/CulturaX \
    --dataset_config_name cs \
    --dataset_data_files cs/cs_part_00000.parquet,cs/cs_part_00001.parquet,cs/cs_part_00002.parquet,cs/cs_part_00003.parquet,cs/cs_part_00004.parquet,cs/cs_part_00005.parquet,cs/cs_part_00006.parquet,cs/cs_part_00007.parquet,cs/cs_part_00008.parquet,cs/cs_part_00009.parquet,cs/cs_part_00011.parquet,cs/cs_part_00012.parquet,cs/cs_part_00013.parquet,cs/cs_part_00014.parquet,cs/cs_part_00015.parquet,cs/cs_part_00016.parquet,cs/cs_part_00017.parquet,cs/cs_part_00018.parquet,cs/cs_part_00019.parquet \
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
    --output_dir ./out/${NOW:-0000_000000}/gpt2_cx-cs_00000-00019_50k
