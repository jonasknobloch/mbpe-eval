#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --mincpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=p_scads_nlp
#SBATCH --job-name=morph-gpt_gpt2_tiny-shakespeare

source /software/rapids/r23.10/Anaconda3/2023.07-2/etc/profile.d/conda.sh

conda activate /home/jknobloc/morph-gpt/env

export WANDB_PROJECT="morph-gpt_gpt2_tiny-shakespeare"
export HF_HOME="/home/jknobloc/morph-gpt/.hf"

python run_clm.py \
    --model_type gpt2 \
    --tokenizer_file ./tokenizers/tokenizer_gpt2_tiny-shakespeare.json \
    --token hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \
    --dataset_name tiny_shakespeare \
    --validation_split_percentage 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --num_train_epochs 30.0 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --report_to wandb \
    --output_dir ./out/gpt2_tiny-shakespeare
