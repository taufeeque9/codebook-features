# Codebook Features

Codebook Features is a library for training neural networks along with vector quantization bottlenecks called _codebooks_ that serve as a good fundamental unit of analysis and control for neural networks. The library provides a range of features to intrepret a trained codebook model like analysing the activations of codes, searching for codes that activate on a pattern, and performing code interventions to verify the causal effect of a code on the output of a model. Many of these features are also available through an easy-to-use webapp that helps in analysing and experimenting with the codebook models.

## Installation
Create a virtual environment and then:
```
git clone https://github.com/alextamkin/codebook-features
cd codebook-features
pip install -e .
```

## Usage

### Training a codebook model

We use the [hydra](https://hydra.cc/) library for configuration management of the training scripts. The default config for training codebooks is available in `codebook_features/config/main.yaml`. The hydra syntax can be used to override any of the default config values. For example, to train a codebook model using gpt2-small on the wikitext dataset, run:
```
python -m codebook_features.train_codebook model_args.model_name_or_path=roneneldan/TinyStories-1M 'data_args.dataset_name=roneneldan/TinyStories'
```

### Interpretability WebApp for Codebook Models

Once a codebook model has been trained and saved on disk, we can use the interpretability webapp to visualize the codebook. First, we need to generate the relevant cache files for the codebook model that is required for the webapp. This can be done by running the script `codebook_features/code_search_cache.py`:
```
python -m codebook_features.code_search_cache --model_name <path to codebook model> --pretrained_path --dataset_name <dataset name> --dataset_config_name <dataset config name> --output_base_dir <path to output directory>
```

Once the cache files have been generated, we can run the webapp using the following command:
```
python -m streamlit run codebook_features/webapp/Code_Browser.py -- --cache_dir <path to cache directory>
```

### Code Intervention

For a general tutorial on using codebook models and seeing how you can perform code intervention, please see the [Code Intervention Tutorial]().

## Guide

### Codebook Model

`codebook_features/models` is the main module used to define codebooks. It has the following classes:
- `CodebookLayer`: defines a `torch.nn.Module` that implements the codebook layer. It takes in arguments like `num_codes`, `dim`, `snap_fn` `kcodes` that define the codebook. It provides various functionalities including logging methods, hook function that can disable specific codes during inference, etc.
  - `GroupCodebookLayer`: defines a `torch.nn.Module` that implements a group of codebook layer each of which are applied to a different part of the input vector. This is useful for applying a group of codebooks on the attention head outputs of a transformer model.
- `CodebookWrapper`: is an abstract class to wrap a codebook around any `torch.nn.Module`. It takes in the `module_layer`, `codebook_cls`, and arguments for the codebook class to instantiate the codebook layer. The wrapper provides a `snap` boolean field that can be used to enable/disable the codebook layer.
  - `TransformerLayerWrapper`: subclasses `CodebookWrapper` to wrap a codebook around a transformer layer, i.e. a codebook is applied on the output of the a whole transformer block.
  - `MLPWrapper`: subclasses `CodebookWrapper` to wrap a codebook around an MLP layer, i.e. a codebook is applied on the output of the MLP block.
- `CodebookModelConfig`: defines the config to be used by a codebook model. It contains important parameters like `codebook_type`, `num_codes`, `num_codebooks`, `layers_to_snap`, `similarity_metric`, `codebook_at`, etc.
- `CodebookModel`: defines the abstract base class for a codebook model. It takes in a neural network model through the `model` argument and the config through the `config` argument and return a codebook model.
  - `GPT2CodebookModel`: subclasses `CodebookModel` to define a codebook model specifically for GPT2.
  - `GPTNeoCodebookModel`: subclasses `CodebookModel` to define a codebook model specifically for GPTNeo.
  - `GPTNeoXCodebookModel`: subclasses `CodebookModel` to define a codebook model specifically for GPTNeoX.
  - `HookedTransformerCodebookModel`: subclasses `CodebookModel` to define a codebook model for any transformer model defined using the `HookedTransformer` class of `transformer_lens`. This is mostly while interpreting the codebooks while the other classes are used for training the codebook models. The `convert_to_hooked_model()` function can be used to convert a trained codebook model to a `HookedTransformerCodebookModel`.

### Codebook Training
The `codebook_features/train_codebook.py` script is used to train a codebook model based on a causal language model. We use the `run_clm.py` script provided by the transformers library for training. It can take in a dataset name available in the [datasets](https://huggingface.co/datasets) library or a custom dataset. The default arguments for the training script is available in `codebook_features/config/main.yaml`. The hydra syntax can be used to override any of the default config values.

### Toy Experiments
The `codebook_features/train_toy_model.py` script provides an algorithmic sequence modeling task to analyse the codebook models. The task is to predict the next element in a sequence of numbers generated using a Finite State Automata (FSM). The `train_toy_model/ToyGraph` class defines the FSM by taking in the number of states through `N`, the number of outbound edges from each state through `edges`, and the base in which to represent the state using `representation_base`. The `train_toy_model/ToyDataset` class defines an iterable torch dataset using the FSM that generates the dataset on the fly. The `train_toy_model/ToyModelTrainer` provides additional logging feature specific to the toy example like logging the transition accuracy of a model.

The `codebook_features/train_toy_model.py` script can be used to train a codebook model on the toy dataset. The syntax for the arguments and training procedure is similar to the `train_codebook.py` script. The default arguments for the training script is available in `codebook_features/config/toy_main.yaml`.



For tutorials on how to use the library, please see the [Codebook Features Tutorials]().


## Citations (BibTeX)

```
@misc{codebookfeatures,
  author = {},
  title = {},
  year = {2023},
  howPublished = {},
  archivePrefix = {arXiv},
  eprint = {},
  primaryClass = {},
  url = {},
}
```
