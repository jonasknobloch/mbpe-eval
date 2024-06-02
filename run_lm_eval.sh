#!/usr/bin/env bash

source /software/rapids/r23.10/Anaconda3/2023.07-2/etc/profile.d/conda.sh

conda activate /home/jknobloc/cbt-eval/env

export HF_HOME="/home/jknobloc/cbt-eval/.hf"

lm_eval --model_args pretrained=/home/jknobloc/morph-gpt/out/2024-04-16_090023/gpt2+morf_s0-30-x-2_cx-en_00000-00009_50k/checkpoint-290000 --task swag --device cuda

lm_eval --model hf --model_args pretrained=jonasknobloch/gpt2_cx-en_00000-00009_50k --task swag --device cuda --num_fewshot 5
lm_eval --model hf --model_args pretrained=jonasknobloch/gpt2-ts_cx-en_00000-00009_50k --task swag --device cuda
lm_eval --model hf --model_args pretrained=jonasknobloch/gpt2-morf_u0-30-50-x_cx-en_00000-00009_50k --task swag --device cuda
lm_eval --model hf --model_args pretrained=jonasknobloch/gpt2-morf_s0-30-x-2_cx-en_00000-00009_50k --task swag --device cuda

# hf (pretrained=jonasknobloch/gpt2_cx-en_00000-00009_50k), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
# |Tasks|Version|Filter|n-shot| Metric |Value |   |Stderr|
# |-----|------:|------|-----:|--------|-----:|---|-----:|
# |swag |      1|none  |     0|acc     |0.3581|±  |0.0034|
# |     |       |none  |     0|acc_norm|0.4544|±  |0.0035|

# hf (pretrained=jonasknobloch/gpt2-ts_cx-en_00000-00009_50k), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
# |Tasks|Version|Filter|n-shot| Metric |Value |   |Stderr|
# |-----|------:|------|-----:|--------|-----:|---|-----:|
# |swag |      1|none  |     0|acc     |0.3561|±  |0.0034|
# |     |       |none  |     0|acc_norm|0.4454|±  |0.0035|

# hf (pretrained=jonasknobloch/gpt2-morf_u0-30-50-x_cx-en_00000-00009_50k), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
# |Tasks|Version|Filter|n-shot| Metric |Value |   |Stderr|
# |-----|------:|------|-----:|--------|-----:|---|-----:|
# |swag |      1|none  |     0|acc     |0.3549|±  |0.0034|
# |     |       |none  |     0|acc_norm|0.4425|±  |0.0035|

# hf (pretrained=jonasknobloch/gpt2-morf_s0-30-x-2_cx-en_00000-00009_50k), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
# |Tasks|Version|Filter|n-shot| Metric |Value |   |Stderr|
# |-----|------:|------|-----:|--------|-----:|---|-----:|
# |swag |      1|none  |     0|acc     |0.3514|±  |0.0034|
# |     |       |none  |     0|acc_norm|0.4442|±  |0.0035|

# hf (pretrained=jonasknobloch/gpt2_cx-en_00000-00009_50k), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 1
# |Tasks|Version|Filter|n-shot| Metric |Value |   |Stderr|
# |-----|------:|------|-----:|--------|-----:|---|-----:|
# |swag |      1|none  |     5|acc     |0.3516|±  |0.0034|
# |     |       |none  |     5|acc_norm|0.4406|±  |0.0035|

# hf (pretrained=/home/jknobloc/morph-gpt/out/2024-04-16_090023/gpt2+morf_s0-30-x-2_cx-en_00000-00009_50k/checkpoint-290000), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
# |Tasks|Version|Filter|n-shot| Metric |Value |   |Stderr|
# |-----|------:|------|-----:|--------|-----:|---|-----:|
# |swag |      1|none  |     0|acc     |0.3516|±  |0.0034|
# |     |       |none  |     0|acc_norm|0.4445|±  |0.0035|
