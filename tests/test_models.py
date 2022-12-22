"""Tests for model classes."""

import transformers

from codebook_features import models


def test_codebook_model():
    c = transformers.BertConfig()
    tokenizer = transformers.BertTokenizer.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity"
    )
    m = transformers.BertForSequenceClassification(c)
    mp = models.CodebookModel(m)
    # assert m is not modified
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    output = mp(**inputs)
    assert output is not None


# add test to check if straight through gradient is flowing correctly
