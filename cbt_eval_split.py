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


def split_question(sample):
    question = sample["question"]
    question_detokenized = detokenizer.detokenize(question.split())

    prefix, suffix = question_detokenized.split("XXXXX", 1)

    return prefix, suffix


def get_continuation_log_probability(context, continuation):
    prompt = context + continuation

    context_ids = tokenizer.encode(context, return_tensors="pt")
    continuation_ids = tokenizer.encode(continuation, return_tensors="pt")
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

    # l_ctx = len(context_ids[0])
    # l_con = len(continuation_ids[0])
    # l_pro = len(prompt_ids[0])

    # start of continuation might be tokenized with whitespace at the end of context
    # i is index of last token before continuation tokens so i+1 is index of first continuation token
    # probability for i+1 token found in probability distribution of i-th token
    # most likely next token found by argmax of probability distribution of last token
    for i in range(len(prompt_ids[0]) - len(continuation_ids[0]) - 1, len(prompt_ids[0]) - 1):
        # token = prompt_ids[0][i + 1]
        # token_decoded = tokenizer.decode(token)

        # add log probability i+1 th token i.e. the next predicted token
        total_log_probability += log_probabilities[0][i][prompt_ids[0][i + 1]]

    # TODO options might need a different amount of tokens which skews total log probabilities

    # torch.exp(total_log_probability) for probability
    return total_log_probability


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="jonasknobloch/gpt2_cx-en_00000-00009_50k")
    parser.add_argument("--dataset_path", type=str, default="cbt")
    parser.add_argument("--dataset_name", type=str, default="NE")
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

        question_prefix, question_suffix = split_question(sample)

        options = sample["options"]

        likelihoods = torch.zeros(len(options))

        for i, option in enumerate(options):
            likelihoods[i] = get_continuation_log_probability(context + question_prefix, option + question_suffix)

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
