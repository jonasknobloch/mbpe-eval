#!/usr/bin/env bash

source /software/rapids/r23.10/Anaconda3/2023.07-2/etc/profile.d/conda.sh

conda activate /home/jknobloc/cbt-eval/env

export HF_HOME="/home/jknobloc/cbt-eval/.hf"

python cbt_eval.py --model_path jonasknobloch/gpt2_cx-en_00000-00009_50k --dataset_name CN
python cbt_eval.py --model_path jonasknobloch/gpt2_cx-en_00000-00009_50k --dataset_name NE
python cbt_eval.py --model_path jonasknobloch/gpt2_cx-en_00000-00009_50k --dataset_name P
python cbt_eval.py --model_path jonasknobloch/gpt2_cx-en_00000-00009_50k --dataset_name V

python cbt_eval.py --model_path jonasknobloch/gpt2-ts_cx-en_00000-00009_50k --dataset_name CN
python cbt_eval.py --model_path jonasknobloch/gpt2-ts_cx-en_00000-00009_50k --dataset_name NE
python cbt_eval.py --model_path jonasknobloch/gpt2-ts_cx-en_00000-00009_50k --dataset_name P
python cbt_eval.py --model_path jonasknobloch/gpt2-ts_cx-en_00000-00009_50k --dataset_name V

python cbt_eval.py --model_path jonasknobloch/gpt2-morf_u0-30-50-x_cx-en_00000-00009_50k --dataset_name CN
python cbt_eval.py --model_path jonasknobloch/gpt2-morf_u0-30-50-x_cx-en_00000-00009_50k --dataset_name NE
python cbt_eval.py --model_path jonasknobloch/gpt2-morf_u0-30-50-x_cx-en_00000-00009_50k --dataset_name P
python cbt_eval.py --model_path jonasknobloch/gpt2-morf_u0-30-50-x_cx-en_00000-00009_50k --dataset_name V

python cbt_eval.py --model_path jonasknobloch/gpt2-morf_s0-30-x-2_cx-en_00000-00009_50k --dataset_name CN
python cbt_eval.py --model_path jonasknobloch/gpt2-morf_s0-30-x-2_cx-en_00000-00009_50k --dataset_name NE
python cbt_eval.py --model_path jonasknobloch/gpt2-morf_s0-30-x-2_cx-en_00000-00009_50k --dataset_name P
python cbt_eval.py --model_path jonasknobloch/gpt2-morf_s0-30-x-2_cx-en_00000-00009_50k --dataset_name V
