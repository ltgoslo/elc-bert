### Preprocess Folder

This folder contains all the scripts to preprocess the raw BabyLM data. While there are a few scripts, the whoe preprocessing can be done running:

```bash
./run.sh
```

Make sure that you have python as well as the following packages:

 - `smart-open`
 - `six`
 - `sacremoses`
 - `ftfy`
 - `nltk`

After running `run.sh` you will have one processed file for each dataset, as well as two additional files:

 - `all.txt`: All the datasets concatenated, useful to create your tokenizer.
 - `segmented.txt`: The `all.txt` file plit into sentences, useful to cache the dataset.

By default all of those files are saved in `../data/processed/`.