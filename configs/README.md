## Configs folder

This folder contains the model config files. Here is an example configuration file:

base.json
```json
{
    "attention_probs_dropout_prob": 0.1,
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "intermediate_size": 2048,
    "max_position_embeddings": 512,
    "position_bucket_size": 32,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "vocab_size": 16384,
    "layer_norm_eps": 1.0e-7
  }
```

To add a new model parameter simply put a comma to the last line and add on the next line:

```json
    "PARAM_NAME": PARAM_VALUE
```