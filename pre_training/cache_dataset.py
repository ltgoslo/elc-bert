from tokenizers import Tokenizer
from smart_open import open
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Cached Dataset Creation')
parser.add_argument(
    '--segments_path',
    type=str,
    default="../data/processed/segmented.txt",
    help='Path to the segmented data file.'
)
parser.add_argument(
    '--tokenizer_path',
    type=str,
    default="../tokenizers/tokenizer.json",
    help='Path to the tokenizer JSON file.'
)
parser.add_argument(
    '--sequence_length',
    type=int,
    default=128,
    help='Sequence length of each cached input sequence.'
)
args = parser.parse_args()


SEQ_LEN = args.sequence_length - 2
tokenizer = Tokenizer.from_file(args.tokenizer_path)


documents = [[]]
for line in tqdm(open(args.segments_path)):
    line = line.strip()

    if len(line) == 0:
        if len(documents[-1]) > 0:
            documents.append([])
        continue

    ids = tokenizer.encode(line, add_special_tokens=False).ids
    documents[-1].append(ids)


with open(f"../../data/processed/cached_{SEQ_LEN + 2}.txt", "w") as f:
    for document in tqdm(documents):
        segment = []
        for i, sentence in enumerate(document):
            segment += sentence

            if len(segment) > SEQ_LEN:
                segment = segment[:SEQ_LEN]
                subwords = [
                    tokenizer.id_to_token(token_id) for token_id in segment
                ]
                f.write(" ".join(subwords) + "\n")

                segment = [s for s in sentence]

        if len(segment) > 0:
            segment = segment[:SEQ_LEN]
            subwords = [
                tokenizer.id_to_token(token_id) for token_id in segment
            ]
            f.write(" ".join(subwords) + "\n")
