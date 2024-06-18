<h2 align="center"><b><h3>Not all layers are equally as important:</h3><h3>Every Layer Counts BERT</h3></b></h2><br>


<p align="center">
  <b>Lucas Georges Gabriel Charpentier and David Samuel</b>
</p>

<p align="center">
  <i>
    University of Oslo<br>
    Language Technology Group<br>
  </i>
</p>
<br>

<p align="center">
  <a href="https://aclanthology.org/2023.conll-babylm.20/"><b>Paper</b></a><br>
  <a href="https://huggingface.co/lgcharpe/ELC_BERT_baby_100M"><b>HuggingFace 100M model</b></a><br>
  <a href="https://huggingface.co/lgcharpe/ELC_BERT_small_baby_10M"><b>HuggingFace 10M model</b></a>
</p>

_______

<br>

<h3 align="center"><b>Abstract</b></h3><br>

This paper introduces a novel modification of
the transformer architecture, tailored for the
data-efficient pretraining of language models.
This aspect is evaluated by participating in the
BabyLM challenge, where our solution won
both the STRICT and STRICT-SMALL tracks.
Our approach allows each transformer layer to
select which outputs of previous layers to pro-
cess. The empirical results verify the potential
of this simple modification and show that not
all layers are equally as important.

_______

<br>

This is the official repository for our BabyLM 2023 submission: ELC-BERT.

_______

<br>

## Content of this repository

- `./train_elc_bert_*.py`: Scripts to train an ELC-BERT model (replace * with base, normalized, weighted_output, or zero).
- `./preprocess/`: Scripts for processing the BabyLM 2023 datasets.
- `./tokenizers/`: Script for creating a tokenizer as well as where the tokenizers are saved.
- `./configs/`: Folder containing model configs.
- `./pre_training/`: Scripts for the dataset, optimizer and utilities of pretraining.
- `./models/`: Folder containing training models.

_______

<br>

## Code to pre-train a model

1. Run `preprocess/run.sh`.
2. Run `tokenizers/create_tokenizers.py` or use one of the provided tokenizers in the tokenizers folder.
3. Run `pre_training/cache_dataset.py`.
4. Choose a config file found in configs, or create your own config file in the same style.
6. Run one of the `train_elc_bert_*.py`, your pre-trained model will be found in the models folder.

Specific details on the files and scripts can be found in the READMEs and scripts themselves.
_______

<br>

## Training

After preprocessing your data, creating your tokenizer, and caching the data with your tokenizer, you are ready to train your ELC BERT model. To this extent, you can run:

```bash
python train_elc_bert_*.py \
    --input_path="PATH_TO_CACHED_DATA" \
    --config_file="PATH_TO_CONFIG_FILE" \
    --output_dir="PATH_TO_OUTPUT_DIR" \
    --vocab_path="PATH_TO_TOKENIZER_FILE" \
    --checkpoint_path="PATH_TO_MODEL_CHECKPOINT" \ # (Optional, to continue training)
    --optimizer="NAME_OF_OPTIMIZER" \ # Options: lamb, adamw
    --scheduler="NAME_OF_SCHEDULER" \ # (Not implemented) Options: cosine
    --seq_length=MAX_SEQUENCE_LENGTH \
    --batch_size=TRAINING_BATCH_SIZE \
    --learning_rate=MAX_TRAINING_LAEARNING_RATE \
    --max_steps=NUMBER_OF_TRAINING_STEPS \
    --long_after=FRACTION_AFTER_WHICH_TO_4x_SEQUENCE_LENGTH \
    --warmup_proportion=FRACTION_OF_TRAINING_STEPS_FOR_WARMUP \
    --seed=RANDOMIZATION_SEEd \
    --log_freq=LOSS_LOGGING_FREQUENCY \ # For WANDB, unused
    --mask_p=TOKEN_MASKING_PROBABILITY \
    --short_p=PROBABILITY_OF_SHORTENING_SEQUENCE \
    --weight_decay=FRACTION_OF_WEIGHT_DECAY \
    --max_gradient=MAX_GRADIENT_BEFORE_CLIPPING \
    --gradient_accumulation=NUMBER_GRADIENT_ACCUMULATION_STEPS \
    --label_smoothing=CROSS_ENTROPY_LABEL_SMOOTHING \
    --wandb_entity="WANDB_ENTITY_NAME" \
    --wandb_name="WANDB_RUN_NAME" \
    --wandb_project="WANDB_PROJECT_NAME"
```

A few things to note:
 - In the dataset (look up `pre_training/dataset.py`) you can pass a `random_p` and `keep_p` representing the probability of a masked token being replaced by either a random token or the original token. In the code they are both set to 0.1 by default, but this can be changed.
 - Our code assumes the usage of wandb but this can be removed. In general, before calling wandb we do a check for `is_main_process()` (when running multiple GPUs/CPUs, it makes sure only one process (the main) executes the code) to make sure to not have multiple wandb runs for the same model.
 - We assume the usage of SLURM at the start of the code (to import wandb) (lines 31-32), if you do not use SLURM remove line 31 (and line 32 if you do not use wandb).
<br>

## Please cite the following publication
```bibtex
@inproceedings{georges-gabriel-charpentier-samuel-2023-layers,
    title = "Not all layers are equally as important: Every Layer Counts {BERT}",
    author = "Georges Gabriel Charpentier, Lucas  and
      Samuel, David",
    editor = "Warstadt, Alex  and
      Mueller, Aaron  and
      Choshen, Leshem  and
      Wilcox, Ethan  and
      Zhuang, Chengxu  and
      Ciro, Juan  and
      Mosquera, Rafael  and
      Paranjabe, Bhargavi  and
      Williams, Adina  and
      Linzen, Tal  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.conll-babylm.20",
    doi = "10.18653/v1/2023.conll-babylm.20",
    pages = "238--252",
}
```
