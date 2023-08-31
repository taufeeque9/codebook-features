# Codebook Features

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
python -m codebook_features.train_codebook model_args.model_name_or_path=gpt2 'data_args={dataset_name:wikitext,dataset_config_name:wikitext-103-v1}'
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