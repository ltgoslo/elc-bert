## Tokenizers Folder

This folder contains both the script to create your own tokenizer as well as the created tokenizers. The tokenizers are created as BPE tokenizers. To create your own tokenizer run:

```bash
python create_tokenizer.py \
    --input_path="PATH_TO_DATA_FILE" \ # If you used the preprocess code, this is the all.txt file
    --vocab_path="PATH_TO_WHERE_TO_SAVE_TOKENIZER_FILE"
    --vocab_size=MAX_VOCABULARY_SIZE
    --min_frequency=MINIMUM_FREQUENCY_OF_TOKEN_TO_BE_INCLUDED_IN_VOCABULARY
```

In addition the script will output the frequency of the 95% most frequent token, there is some evidence that having this number around 100 is good, but this is not a definite measure.

We include three tokenizers:

 - `tokenizer.json` created on the 100M dataset with a vocabulary size of 16384. (Used for our strict submission)
 - `tokenizer_small_104.json` created on the 10M dataset with a vocabulary size of 6144. (Used for our small-strict submission)
 - `tokenizer_small_68.json` created on the 10M dataset with a vocabulary size of 8192.