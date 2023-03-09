"""Tests for model classes."""

import pytest
import torch
import transformers

from codebook_features import evaluation, models, run_clm


def test_bert_codebook_model():
    config = transformers.BertConfig()
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    model = transformers.BertForSequenceClassification(config)
    codebook_model = models.BertCodebookModel(model, 100, [1, 5, -1])
    # assert m is not modified
    input = tokenizer("Hello, my dog is cute", return_tensors="pt")
    output = codebook_model(**input)
    assert output is not None


def test_gpt_codebook_model():
    config = transformers.GPT2Config()
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    model = transformers.GPT2LMHeadModel(config)
    codebook_model = models.GPT2CodebookModel(model, 100, [1, 5, -1])

    input = tokenizer("Hello, my dog is cute", return_tensors="pt")
    output = codebook_model(**input)
    assert output is not None


def test_codebook_layer():
    layer = models.CodebookLayer(dim=100, num_codes=3)
    input = torch.randn(1, 10, 100)
    output = layer(input)
    assert output is not None


@pytest.mark.parametrize(
    "codebook_cls",
    [models.CompositionalCodebookLayer2, models.CompositionalCodebookLayer],
)
@pytest.mark.parametrize("num_codebooks", [1, 8])
def test_composition_codebook_layer(codebook_cls, num_codebooks):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    layer = codebook_cls(dim=768, num_codes=1000, num_codebooks=num_codebooks)
    layer.to(device)
    input = torch.randn(16, 128, 768, device=device)
    output = layer(input)
    assert output.shape == input.shape


def test_evaluate():
    config = models.CodebookModelConfig()
    model_args = run_clm.ModelArguments(model_name_or_path="taufeeque/tiny-gpt2")
    model = models.wrap_codebook(
        model_or_path=model_args.model_name_or_path, config=config
    )
    data_args = run_clm.DataTrainingArguments(
        dataset_name="wikitext",
        dataset_config_name="wikitext-103-v1",
        streaming=False,
    )
    tokens, cb_acts, metrics = evaluation.evaluate(
        model=model, model_args=model_args, data_args=data_args, eval_on="validation"
    )
    for _, v in cb_acts.items():
        assert len(v) == len(tokens)
    assert metrics is not None


def test_hooked_transformer_codebook_model():
    config = models.CodebookModelConfig(
        layers_to_snap="all", codebook_at="attention", codebook_type="compositional"
    )
    model_path = "EleutherAI/pythia-70m-deduped"
    model_args = run_clm.ModelArguments(model_name_or_path=model_path)
    model = models.wrap_codebook(
        model_or_path=model_args.model_name_or_path, config=config
    )
    hooked_model = models.convert_to_hooked_model(model_path, model)
    assert hooked_model is not None
