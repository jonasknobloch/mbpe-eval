import argparse

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="jonasknobloch/gpt2_cx-en_00000-00009_50k")
    parser.add_argument("--dataset_path", type=str, default="uonlp/CulturaX")
    parser.add_argument("--dataset_name", type=str, default="en")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_files", type=str, default="en/en_part_00010.parquet")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--stride", type=int, default=1024)

    args = parser.parse_args()

    global tokenizer, model

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)

    samples = load_dataset(
        args.dataset_path,
        args.dataset_name,
        split=args.dataset_split,
        data_files=args.dataset_files,
    )

    samples = samples.select(range(args.num_samples))

    encodings = tokenizer("\n\n".join(samples["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = args.stride
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())

    print(f"model: {args.model_path} dataset: {args.dataset_path} {args.dataset_files} num_samples: {args.num_samples} stride: {args.stride} perplexity: {ppl.item()}")


if __name__ == "__main__":
    main()
