from functools import partial
from typing import Optional

import pytest
import torch
import torch as t
from codebook_features import models
from codebook_features.models import HookedTransformerCodebookModel
from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer

from cb_activation_patching import patch_attn_codebook
from more_tl_mods import get_act_name

## Set global device variable
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

## Loading and setting up hooked codebook model
model_name_or_path = "EleutherAI/pythia-410m-deduped"
pre_trained_path = "taufeeque/pythia410m_wikitext_attn_codebook_model"

original_cb_model = models.wrap_codebook(
    model_or_path=model_name_or_path, pretrained_path=pre_trained_path
)

orig_cb_model = original_cb_model.to(DEVICE).eval()

hooked_kwargs = dict(
    center_unembed=False,
    fold_value_biases=False,
    center_writing_weights=False,
    fold_ln=False,
    refactor_factored_attn_matrices=False,
    device=DEVICE,
)

cb_model = models.convert_to_hooked_model(
    model_name_or_path, orig_cb_model, hooked_kwargs=hooked_kwargs
)
cb_model = cb_model.to(DEVICE).eval()
tokenizer = cb_model.tokenizer

orig_tokens = t.ones((5, 10), dtype=t.long, device=DEVICE)
new_tokens = t.zeros((5, 10), dtype=t.long, device=DEVICE)
_, orig_cache = cb_model.run_with_cache(orig_tokens)
_, new_cache = cb_model.run_with_cache(new_tokens)


def extract_cache_from_patched_model(
    cb_model: HookedTransformer,
    codebook_layer: int,
    codebook_head_idx: int,
    position: int,
    orig_tokens: Float[Tensor, "batch pos d_model"],
    new_tokens: Float[Tensor, "batch pos d_model"],
    new_cache: Optional[ActivationCache] = None,
) -> Float[Tensor, "batch pos d_vocab"]:
    """
    Extracts the cache from a patched model, we use this to test that our activation patching
    is working as expected (ie. caches that should be patched over are and caches prior are left the same
    """
    if new_cache is None:
        _, new_cache = cb_model.run_with_cache(new_tokens)
    activation_name = get_act_name(f"cb_{codebook_head_idx}", codebook_layer, "attn")
    hook_fn = partial(patch_attn_codebook, position=position, new_cache=new_cache)
    cb_model.add_hook(name=activation_name, hook=hook_fn, dir="fwd")
    _, modified_cache = cb_model.run_with_cache(orig_tokens)
    cb_model.reset_hooks()
    return modified_cache


def test_cache_patching():
    codebook_layer_to_change = 4
    codebook_head_to_change = 2
    modified_cache = extract_cache_from_patched_model(
        cb_model=cb_model,
        codebook_layer=codebook_layer_to_change,
        codebook_head_idx=codebook_head_to_change,
        position=-1,
        orig_tokens=orig_tokens,
        new_tokens=new_tokens,
    )

    # All caches before the codebook change should be the same
    for layer_idx in range(codebook_layer_to_change):
        for codebook_head_idx in range(cb_model.cfg.n_heads):
            assert t.equal(
                modified_cache[
                    get_act_name(f"cb_{codebook_head_idx}", layer_idx, "attn")
                ],
                orig_cache[get_act_name(f"cb_{codebook_head_idx}", layer_idx, "attn")],
            )

    # The codebook cache that got changed should equal the new cache for the final position
    assert t.equal(
        new_cache[get_act_name("cb_2", 4, "attn")][:, -1],
        modified_cache[get_act_name("cb_2", 4, "attn")][:, -1],
    )
