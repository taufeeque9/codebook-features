"""Tests for model classes."""

import transformers

from codebook_features import models


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


# add test to check if straight through gradient is flowing correctly
