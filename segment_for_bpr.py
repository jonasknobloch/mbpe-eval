from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("./tokenizers/tokenizer_gpt2_tiny-shakespeare.json")

# tokenizer = Tokenizer.from_file("data/tokenizer_gpt2_cx-en_00000-00000.json")
# tokenizer = Tokenizer.from_file("data/tokenizer_gpt2+morf_s0-30-x-2_cx-en_00000-00000_50k.json")
# tokenizer = Tokenizer.from_file("data/tokenizer_gpt2+morf_u0-30-50-x_cx-en_00000-00000_50k.json")
# tokenizer = Tokenizer.from_file("data/tokenizer_gpt2+ts_cx-en_00000-00000_50k.json")

with open('goldstd_trainset.segmentation.eng', 'r') as f:
    with open('ts.bpe.eng', 'w') as o:
        for line in f.readlines():
            word = line.split("\t")[0]
            encoded = tokenizer.encode(word).ids
            segmented = []

            for id in encoded:
                segmented.append(tokenizer.decode([id]))

            # print(word, ' '.join(segmented), sep="\t")
            o.write(word + "\t" + ' '.join(segmented) + "\n")
