from datasets import load_dataset
from tokenizers import Tokenizer

dataset = load_dataset(
    "uonlp/CulturaX",
    "en",
    split="train",
    data_files=["en/en_part_00000.parquet",],
    cache_dir="./data"
)

# dataset = load_dataset(
#     "uonlp/CulturaX",
#     "cs",
#     split="train",
#     data_files=["cs/cs_part_00000.parquet", "cs/cs_part_00001.parquet",],
#     cache_dir="./data"
# )

tokenizer = Tokenizer.from_file("./tokenizers/tokenizer_gpt2_tiny-shakespeare.json")

# tokenizer = Tokenizer.from_file("./tokenizers/tokenizer_gpt2_cx-en_00000-00000.json")
# tokenizer = Tokenizer.from_file("./tokenizers/tokenizer_gpt2_cx-en_00000-00000_50k.json")
# tokenizer = Tokenizer.from_file("./tokenizers/tokenizer_gpt2+ts_cx-en_00000-00000_50k.json")
# tokenizer = Tokenizer.from_file("./tokenizers/tokenizer_gpt2+morf_cx-en_00000-00000_50k.json")
# tokenizer = Tokenizer.from_file("./tokenizers/tokenizer_gpt2+morf_s0-30-x-2_cx-en_00000-00000_50k.json")

# tokenizer = Tokenizer.from_file("./tokenizers/tokenizer_gpt2_cx-cs_00000-00001.json")
# tokenizer = Tokenizer.from_file("./tokenizers/tokenizer_gpt2_cx-cs_00000-00001_50k.json")
# tokenizer = Tokenizer.from_file("./tokenizers/tokenizer_gpt2+ts_cx-cs_00000-00001_50k.json")

words = 0
tokens = 0

for sample in dataset:
    if sample['text'].strip() == '':
        continue

    words += len(sample['text'].split())
    tokens += len(tokenizer.encode(sample['text']).ids)

print("words:", words)
print("tokens:", tokens)
print("fertility:", tokens/words)

# tokenizer_gpt2_cx-en_00000-00000
# words: 652055294
# tokens: 893453000
# fertility: 1.3702104840207003

# tokenizer_gpt2_cx-en_00000-00000_50k
# words: 652055294
# tokens: 860785098
# fertility: 1.3201105886581452

# tokenizer_gpt2+ts_cx-en_00000-00000_50k
# words: 652055294
# tokens: 910259044
# fertility: 1.3959844393196508

# tokenizer_gpt2+morf_cx-en_00000-00000_50k
# words: 652055294
# tokens: 925433031
# fertility: 1.4192554519770528

# tokenizer_gpt2+morf_s0-30-x-2_cx-en_00000-00000_50k
# words: 652055294
# tokens: 1104039548
# fertility: 1.6931685980606423

# tokenizer_gpt2_cx-cs_00000-00001
# words: 633593583
# tokens: 1044297453
# fertility: 1.6482134305328027

# tokenizer_gpt2_cx-cs_00000-00001_50k
# words: 633593583
# tokens: 976498741
# fertility: 1.5412068038574185

# tokenizer_gpt2+ts_cx-cs_00000-00001_50k
# words: 633593583
# tokens: 998471785
# fertility: 1.5758868331215405
