"""Tests for model classes."""


import pytest
import torch
import transformers

from codebook_features import evaluation, models, run_clm


@pytest.fixture()
def hooked_model(request):
    """Fixture for hooked model."""
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
    if hooked_model.device.type == "mps":
        # codebook layer is not supported on MPS
        hooked_model.cfg.device = torch.device("cpu")
        hooked_model = hooked_model.to(hooked_model.cfg.device)
    return hooked_model


def test_code_replacement():
    """Test code replacement."""
    dim = 100
    layer = models.CodebookLayer(dim=dim, num_codes=10, key="testlayer")
    layer.counts = torch.tensor([1, 0, 3, 0, 2, 0, 0, 0, 0, 0])
    orig_weight = layer.codebook.weight.clone()
    input_tensor = torch.randn(1, 10, dim)
    layer.replace_dead_codes(input_tensor)
    new_weight = layer.codebook.weight
    unreplaced_codes = torch.where(layer.counts > 0)[0]
    assert torch.allclose(new_weight[unreplaced_codes], orig_weight[unreplaced_codes])


def test_codebook_layer():
    """Test CodebookLayer."""
    layer = models.CodebookLayer(dim=100, num_codes=3, key="testlayer")
    input = torch.randn(1, 10, 100)
    output = layer(input)
    assert output is not None


@pytest.mark.parametrize(
    "codebook_cls",
    [models.GroupCodebookLayer],
)
@pytest.mark.parametrize("num_codebooks", [1, 8])
def test_composition_codebook_layer(codebook_cls, num_codebooks):
    """Test GroupCodebookLayer."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dim, num_codes, bs, seq_len = 32, 128, 2, 10
    layer = codebook_cls(
        dim=dim,
        num_codes=num_codes,
        num_codebooks=num_codebooks,
        key="testlayer",
    )
    layer.to(device)
    input = torch.randn(bs, seq_len, dim, device=device)
    output = layer(input)
    assert output.shape == input.shape


def test_evaluate():
    """Test evaluation."""
    config = models.CodebookModelConfig()
    model_args = run_clm.ModelArguments(model_name_or_path="taufeeque/tiny-gpt2")
    model = models.wrap_codebook(
        model_or_path=model_args.model_name_or_path, config=config
    )
    data_args = run_clm.DataTrainingArguments(
        dataset_name="wikitext",
        dataset_config_name="wikitext-103-v1",
        streaming=False,
        max_eval_samples=2,
    )
    tokens, cb_acts, metrics, _ = evaluation.evaluate(
        model=model,
        model_args=model_args,
        data_args=data_args,
        eval_on="validation",
        tf32=False,
    )
    for _, v in cb_acts.items():
        assert len(v) == len(tokens)
    assert metrics is not None


def compare_input_on_hooked_model(input, orig_model, hooked_model):
    """Compare input on original and hooked model."""
    if isinstance(input, str):
        input = hooked_model.tokenizer(input, return_tensors="pt")["input_ids"]
    if torch.cuda.is_available():
        orig_model = orig_model.cuda()
        hooked_model = hooked_model.cuda()
        input = input.cuda()

    orig_output = orig_model(input)["logits"]
    hooked_output = hooked_model(input)
    assert torch.allclose(orig_output, hooked_output)


@pytest.mark.parametrize("codebook_at", ["attention", "preproj_attention", "mlp"])
def test_hooked_transformer_codebook_model(codebook_at):
    """Test HookedTransformerCodebookModel."""
    config = models.CodebookModelConfig(
        layers_to_snap="all",
        codebook_at=codebook_at,
        codebook_type="group",
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
            "fold_value_biases": False,
            "refactor_factored_attn_matrices": False,
        },
    )
    if hooked_model.device.type == "mps":
        # codebook layer is not supported on MPS
        hooked_model.cfg.device = torch.device("cpu")
        hooked_model = hooked_model.to(hooked_model.cfg.device)

    sentence = "this is a random sentence to test."
    input = hooked_model.model.tokenizer(sentence, return_tensors="pt")["input_ids"]
    if torch.cuda.is_available():
        model = model.cuda()
        hooked_model = hooked_model.cuda()
        input = input.cuda()
    output = model(input)["logits"]
    hooked_output = hooked_model(input)

    assert torch.allclose(output, hooked_output, rtol=1e-4)


@pytest.mark.parametrize(
    "hooked_model", [{"k_codebook": 10, "num_codes": 15}], indirect=True
)
def test_hook_kwargs_not_keep_k_codes(hooked_model):
    """Test that when keep_k_codes is False, then the blocked codes are not used for averaging."""
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
    """Test that when all the possible codes are blocked, then the output of the codebook layer is zero."""
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
    """Test that when all the topk codes are blocked, then the output of the codebook layer is zero."""
    disable_for_tkns = "all"
    hooked_model.set_hook_kwargs(
        keep_k_codes=False,
        disable_for_tkns=disable_for_tkns,
        disable_topk=hooked_model.config.k_codebook[0],
    )
    sentence = "this is a random sentence to test."
    input = hooked_model.model.tokenizer(sentence, return_tensors="pt")["input_ids"]
    _, cache = hooked_model.run_with_cache(input)
    for k, v in cache.items():
        if "codebook_ids" in k:
            assert torch.isin(v, -1).all().item()
        elif "hook_mlp_out" in k:
            assert torch.isin(v, 0).all().item()
            assert torch.isin(v, 0).all().item()


def test_codebook_with_faiss():
    """Test the FaissSnapFn class."""
    try:
        import faiss  # noqa: F401
    except ImportError:
        pytest.skip("Faiss not installed.")
    model_path = "taufeeque/tiny-gpt2"
    config = models.CodebookModelConfig(layers_to_snap="all", k_codebook=10)
    model_args = run_clm.ModelArguments(model_name_or_path=model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = models.wrap_codebook(
        model_or_path=model_args.model_name_or_path, config=config
    )
    model.use_faiss()
    sentence = "this is a random sentence to test."
    input = tokenizer(sentence, return_tensors="pt")["input_ids"]
    output = model(input)
    assert output is not None
