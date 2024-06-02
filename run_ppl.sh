#!/usr/bin/env bash

source /software/rapids/r23.10/Anaconda3/2023.07-2/etc/profile.d/conda.sh

conda activate /home/jknobloc/cbt-eval/env

export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
export HF_HOME="/home/jknobloc/cbt-eval/.hf"

python calc_perplexity.py --model_path jonasknobloch/gpt2_cx-en_00000-00009_50k --dataset_name en --dataset_files en/en_part_00010.parquet --num_samples 1000 --stride 512
python calc_perplexity.py --model_path jonasknobloch/gpt2-ts_cx-en_00000-00009_50k --dataset_name en --dataset_files en/en_part_00010.parquet --num_samples 1000 --stride 512
python calc_perplexity.py --model_path jonasknobloch/gpt2-morf_u0-30-50-x_cx-en_00000-00009_50k --dataset_name en --dataset_files en/en_part_00010.parquet --num_samples 1000 --stride 512
python calc_perplexity.py --model_path jonasknobloch/gpt2-morf_s0-30-x-2_cx-en_00000-00009_50k --dataset_name en --dataset_files en/en_part_00010.parquet --num_samples 1000 --stride 512

python calc_perplexity.py --model_path jonasknobloch/gpt2_cx-cs_00000-00019_50k --dataset_name cs --dataset_files cs/cs_part_00020.parquet --num_samples 1000 --stride 512
python calc_perplexity.py --model_path jonasknobloch/gpt2-ts_cx-cs_00000-00019_50k --dataset_name cs --dataset_files cs/cs_part_00020.parquet --num_samples 1000 --stride 512

python calc_perplexity.py --model_path jonasknobloch/gpt2_cx-en_00000-00009_50k --dataset_name en --dataset_files en/en_part_00010.parquet --num_samples 1000 --stride 1024
python calc_perplexity.py --model_path jonasknobloch/gpt2-ts_cx-en_00000-00009_50k --dataset_name en --dataset_files en/en_part_00010.parquet --num_samples 1000 --stride 1024
python calc_perplexity.py --model_path jonasknobloch/gpt2-morf_u0-30-50-x_cx-en_00000-00009_50k --dataset_name en --dataset_files en/en_part_00010.parquet --num_samples 1000 --stride 1024
python calc_perplexity.py --model_path jonasknobloch/gpt2-morf_s0-30-x-2_cx-en_00000-00009_50k --dataset_name en --dataset_files en/en_part_00010.parquet --num_samples 1000 --stride 1024

python calc_perplexity.py --model_path jonasknobloch/gpt2_cx-cs_00000-00019_50k --dataset_name cs --dataset_files cs/cs_part_00020.parquet --num_samples 1000 --stride 1024
python calc_perplexity.py --model_path jonasknobloch/gpt2-ts_cx-cs_00000-00019_50k --dataset_name cs --dataset_files cs/cs_part_00020.parquet --num_samples 1000 --stride 1024
