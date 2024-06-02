import argparse
from itertools import islice

import torch
import torch.nn.functional as F
from datasets import load_dataset
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None

detokenizer = TreebankWordDetokenizer()


def build_context(sample):
    sentences = sample["sentences"]

    sentences_detokenized = []

    for sentence in sentences:
        detokenized = detokenizer.detokenize(sentence.split())
        sentences_detokenized.append(detokenized)

    context = " ".join(sentences_detokenized) + " "

    return context


def build_continuations(sample):
    question = sample["question"]
    options = sample["options"]

    continuations = []

    for option in options:
        continuations.append(detokenizer.detokenize(question.replace("XXXXX", option).split()))

    return continuations


def get_continuation_log_probability(context, continuation):
    prompt = context + continuation

    context_ids = tokenizer.encode(context, return_tensors="pt")
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if len(prompt_ids[0]) > 1024:
        tqdm.write(f"Warning: prompt_ids length is {len(prompt_ids[0])}, truncating to 1024 tokens")

        offset = len(prompt_ids[0]) - 1024
        # resize prompt_ids to 1024 tokens
        prompt_ids = prompt_ids[:, offset:]
        # remove offset from context_ids
        context_ids = context_ids[:, offset:]

        if len(prompt_ids[0]) != 1024:
            raise RuntimeError("invalid prompt_ids length")

    with torch.no_grad():
        # single forward pass
        outputs = model(prompt_ids)

    # log softmax logits to get log probabilities / simply taking outputs.logits is not enough
    log_probabilities = F.log_softmax(outputs.logits, dim=-1)

    total_log_probability = torch.zeros(1).to(device)

    # start from context length
    for i in range(len(context_ids[0]) - 1, len(prompt_ids[0]) - 1):
        # add log probability i+1 th token i.e. the next predicted token
        total_log_probability += log_probabilities[0][i][prompt_ids[0][i + 1]]

    # torch.exp(total_log_probability) for probability
    return total_log_probability


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="jonasknobloch/gpt2_cx-en_00000-00009_50k")
    parser.add_argument("--dataset_path", type=str, default="cbt")
    parser.add_argument("--dataset_name", type=str, default="CN")
    parser.add_argument("--dataset_split", type=str, default="validation")

    parser.add_argument("--output_file", type=str)

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = f"{args.model_path.split('/')[-1]}_{args.dataset_path}-{args.dataset_name}-{args.dataset_split}.csv"

    global tokenizer, model

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)

    samples = load_dataset(args.dataset_path, args.dataset_name, split=args.dataset_split)

    num_samples = samples.num_rows
    # num_samples = 100

    true_positives = 0
    # false_negatives = 0

    confidence = torch.tensor(0.0)

    out = open(args.output_file, "w")

    out.write("sample,best_option,answer,accuracy,probability,confidence\n")
    out.flush()

    for n, sample in tqdm(islice(enumerate(samples), num_samples), total=num_samples):
        context = build_context(sample)
        continuations = build_continuations(sample)

        likelihoods = torch.zeros(len(continuations))

        for i, continuation in enumerate(continuations):
            likelihoods[i] = get_continuation_log_probability(context, continuation)

        log_probabilities = F.log_softmax(likelihoods, dim=-1)
        # probabilities = torch.exp(log_probabilities)

        answer = sample["answer"]
        options = sample["options"]

        best = torch.argmax(likelihoods)
        best_option = options[best]

        if best_option == answer:
            true_positives += 1
            confidence += log_probabilities[best]

        # TODO calc distance from best option to answer

        out.write(f"{n},{best_option},{answer},{true_positives / (n + 1)},{torch.exp(log_probabilities[best]).item()},{torch.exp(confidence / true_positives).item()}\n")
        out.flush()


if __name__ == "__main__":
    main()

# validation
# | ~                                          | CBT-CN (acc)      | CBT-NE (acc)      | CBT-P (acc)       | CBT-V (acc)       |
# |--------------------------------------------|-------------------|-------------------|-------------------|-------------------|
# | gpt2_cx-en_00000-00009_50k                 | 0.776 [0.886]     | 0.738 [0.908]     | **0.849** [0.892] | 0.863 [0.914]     |
# | gpt2-ts_cx-en_00000-00009_50k              | **0.780** [0.876] | **0.746** [0.902] | 0.843 [0.883]     | **0.866** [0.905] |
# | gpt2-morf_u0-30-50-x_cx-en_00000-00009_50k | 0.756 [0.884]     | 0.724 [0.903]     | 0.819 [0.878]     | 0.855 [0.897]     |
# | gpt2-morf_s0-30-x-2_cx-en_00000-00009_50k  | 0.728 [0.860]     | 0.732 [0.866]     | 0.810 [0.854]     | 0.825 [0.884]     |

# test
# | ~                                          | CBT-CN (acc)      | CBT-NE (acc)      | CBT-P (acc)       | CBT-V (acc)       |
# |--------------------------------------------|-------------------|-------------------|-------------------|-------------------|
# | gpt2_cx-en_00000-00009_50k                 | **0.765** [0.875] | 0.678 [0.865]     | **0.842** [0.872] | **0.853** [0.903] |
# | gpt2-ts_cx-en_00000-00009_50k              | 0.747 [0.864]     | 0.682 [0.862]     | 0.834 [0.864]     | 0.839 [0.902]     |
# | gpt2-morf_u0-30-50-x_cx-en_00000-00009_50k | 0.739 [0.860]     | 0.661 [0.864]     | 0.818 [0.856]     | 0.838 [0.898]     |
# | gpt2-morf_s0-30-x-2_cx-en_00000-00009_50k  | 0.706 [0.849]     | **0.686** [0.822] | 0.800 [0.842]     | 0.813 [0.876]     |
