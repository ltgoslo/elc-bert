## Pre Training Folder

In this folder you will find scripts for the caching of the dataset, optimizer, dataset and other utility functions

#### `cache_dataset.py`

This script will cache your dataset in to a defined sequence length for slightly faster training.

Usage:

```bash
python cache_dataset.py \
    --segments_path="PATH_TO_SEGMENTED_DATA_FILE" \
    --tokenizer_path="PATH_TO_TOKENIZER_FILE" \
    --sequence_length=SEQUENCE_LENGTH_TO_CACHE_TO
```

#### `config.py`

Script containing the config class. Can import JSON files to create a config object.

#### `dataset.py`

Create the dataset (a Pytorch Dataset object) to use during training. Also has the span masking code. To note, the dataset object expects a cached (tokenized and split in sequence length) file to work. If you want to tokenize on-the-fly, change lines 104-112:

```python
self.segments = []
for i, segment in enumerate(open(file)):
    if i % n_gpus != offset:
        continue

    segment = segment.strip().split(" ")
    assert len(segment) <= seq_length - 2, " ".join(segment)
    segment = [self.tokenizer.token_to_id(token) for token in segment]
    self.segments.append(segment)
```

#### `lamb.py`

Contains the LAMB (You et al., 2019) optimizer code.

#### `utils.py`

Contains a myriad of utilitary functions as well as the learning rate scheduler. Here is a list of all the functions:

 - `cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_factor)`
 - `seed_everything(seed_value)`
 - `get_rank()` (Gets local rank of GPU/CPU)
 - `get_world_size()` (Gets the number of partitions/shards)
 - `is_main_process()` (Checks if local rank is 0 (main))

### References

You, Yang, et al. "Large batch optimization for deep learning: Training bert in 76 minutes." arXiv preprint arXiv:1904.00962 (2019).