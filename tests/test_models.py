"""Tests for model classes."""

import pytest
import torch
import transformers

from codebook_features import evaluation, models, run_clm


@pytest.fixture()
def hooked_model(request):
    config = models.CodebookModelConfig(
        layers_to_snap="all",
        k_codebook=request.param["k_codebook"],
        num_codes=request.param["num_codes"],
    )
    model_path = "EleutherAI/pythia-70m-deduped"
    model_args = run_clm.ModelArguments(model_name_or_path=model_path)
    model = models.wrap_codebook(
        model_or_path=model_args.model_name_or_path, config=config
    )
    hooked_model = models.convert_to_hooked_model(
        model_path=model_path,
        orig_cb_model=model,
    )
    return hooked_model


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


@pytest.mark.parametrize("codebook_at", ["attention", "preproj_attention", "mlp"])
def test_hooked_transformer_codebook_model(codebook_at):
    config = models.CodebookModelConfig(
        layers_to_snap="all",
        codebook_at=codebook_at,
        codebook_type="compositional",
        num_codebooks=-1,
    )
    model_path = "EleutherAI/pythia-70m-deduped"
    model_args = run_clm.ModelArguments(model_name_or_path=model_path)
    model = models.wrap_codebook(
        model_or_path=model_args.model_name_or_path, config=config
    )
    hooked_model = models.convert_to_hooked_model(
        model_path=model_path,
        orig_cb_model=model,
        hooked_kwargs={
            "center_writing_weights": False,
            "center_unembed": False,
            "fold_ln": False,
        },
    )
    sentence = "this is a random sentence to test."
    input = hooked_model.model.tokenizer(sentence, return_tensors="pt")["input_ids"]
    if torch.cuda.is_available():
        model = model.cuda()
        hooked_model = hooked_model.cuda()
        input = input.cuda()
    output = model(input)["logits"]
    hooked_output = hooked_model(input)

    # assert torch.allclose(output, hooked_output)
    assert torch.allclose(output.max(-1).indices, hooked_output.max(-1).indices)


@pytest.mark.parametrize(
    "hooked_model", [{"k_codebook": 10, "num_codes": 15}], indirect=True
)
def test_hook_kwargs_not_keep_k_codes(hooked_model):
    disable_for_tkns = [0, 1, 2]
    disable_codes = [0, 1, 2]
    disable_topk = 3
    hooked_model.set_hook_kwargs(
        keep_k_codes=False,
        disable_for_tkns=disable_for_tkns,
        disable_codes=disable_codes,
        disable_topk=disable_topk,
    )
    sentence = "this is a random sentence to test."
    input = hooked_model.model.tokenizer(sentence, return_tensors="pt")["input_ids"]
    _, cache = hooked_model.run_with_cache(input)
    for k, v in cache.items():
        if "codebook_ids" in k:
            assert (
                not torch.isin(v[:, disable_for_tkns, :], torch.tensor(disable_codes))
                .any()
                .item()
            )
            assert torch.isin(v[:, disable_for_tkns, :disable_topk], -1).all().item()


@pytest.mark.parametrize(
    "hooked_model", [{"k_codebook": 10, "num_codes": 15}], indirect=True
)
def test_hook_kwargs_not_keep_all_codes_returns_zero(hooked_model):
    disable_for_tkns = "all"
    disable_codes = list(range(hooked_model.config.num_codes))
    hooked_model.set_hook_kwargs(
        keep_k_codes=False,
        disable_for_tkns=disable_for_tkns,
        disable_codes=disable_codes,
    )
    sentence = "this is a random sentence to test."
    input = hooked_model.model.tokenizer(sentence, return_tensors="pt")["input_ids"]
    _, cache = hooked_model.run_with_cache(input)
    for k, v in cache.items():
        if "codebook_ids" in k:
            assert torch.isin(v, -1).all().item()
        elif "hook_mlp_out" in k:
            assert torch.isin(v, 0).all().item()


@pytest.mark.parametrize(
    "hooked_model", [{"k_codebook": 10, "num_codes": 15}], indirect=True
)
def test_hook_kwargs_not_keep_all_codes_returns_zero2(hooked_model):
    disable_for_tkns = "all"
    hooked_model.set_hook_kwargs(
        keep_k_codes=False,
        disable_for_tkns=disable_for_tkns,
        disable_topk=hooked_model.config.k_codebook,
    )
    sentence = "this is a random sentence to test."
    input = hooked_model.model.tokenizer(sentence, return_tensors="pt")["input_ids"]
    _, cache = hooked_model.run_with_cache(input)
    for k, v in cache.items():
        if "codebook_ids" in k:
            assert torch.isin(v, -1).all().item()
        elif "hook_mlp_out" in k:
            assert torch.isin(v, 0).all().item()
