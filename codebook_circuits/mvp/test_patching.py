# TODO: Re-write Tests

from functools import partial
from typing import Callable, Optional

import pytest
import torch
import torch as t
from codebook_circuits.mvp.codebook_act_patching import (
    act_patch_attn_codebook,
    basic_codebook_path_patch,
    codebook_activation_patcher,
    patch_attn_codebook_input,
    patch_or_freeze_attn_codebook,
)
from codebook_circuits.mvp.data import TravelToCityDataset
from codebook_circuits.mvp.more_tl_mods import get_act_name
from codebook_circuits.mvp.utils import compute_average_logit_difference
from codebook_features import models
from codebook_features.models import HookedTransformerCodebookModel
from jaxtyping import Float
from torch import Tensor
from transformer_lens import ActivationCache


@pytest.fixture()
def model_setup():
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model_name_or_path = "EleutherAI/pythia-410m-deduped"
    pre_trained_path = "taufeeque/pythia410m_wikitext_attn_codebook_model"
    original_cb_model = models.wrap_codebook(
        model_or_path=model_name_or_path, pretrained_path=pre_trained_path
    )
    orig_cb_model = original_cb_model.to(device).eval()
    hooked_kwargs = dict(
        center_unembed=False,
        fold_value_biases=False,
        center_writing_weights=False,
        fold_ln=False,
        refactor_factored_attn_matrices=False,
        device=device,
    )
    cb_model = models.convert_to_hooked_model(
        model_name_or_path, orig_cb_model, hooked_kwargs=hooked_kwargs
    )
    cb_model = cb_model.to(device).eval()
    return cb_model


@pytest.fixture()
def random_data_setup(model_setup):
    cb_model = model_setup
    device = cb_model.device
    orig_tokens = t.randint(
        0, cb_model.cfg.d_vocab, (5, 10), dtype=t.long, device=device
    )
    new_tokens = t.randint(
        0, cb_model.cfg.d_vocab, (5, 10), dtype=t.long, device=device
    )
    _, orig_cache = cb_model.run_with_cache(orig_tokens)
    _, new_cache = cb_model.run_with_cache(new_tokens)
    return cb_model, orig_tokens, new_tokens, orig_cache, new_cache


@pytest.fixture()
def travel_data_setup(model_setup):
    cb_model = model_setup
    device = cb_model.device
    dataset = TravelToCityDataset(n_examples=10)
    responses = dataset.correct_incorrects
    incorrect_correct_toks = t.cat(
        [cb_model.to_tokens(response, prepend_bos=False).T for response in responses]
    )
    orig_input = dataset.clean_prompts
    new_input = dataset.corrupted_prompts

    orig_tokens = cb_model.to_tokens(orig_input, prepend_bos=True)
    new_tokens = cb_model.to_tokens(new_input, prepend_bos=True)

    _, orig_cache = cb_model.run_with_cache(orig_tokens)
    _, new_cache = cb_model.run_with_cache(new_tokens)

    return (
        cb_model,
        orig_tokens,
        new_tokens,
        incorrect_correct_toks,
        orig_cache,
        new_cache,
    )


def extract_cache_for_run_using_specified_patching_method(
    cb_model: HookedTransformerCodebookModel,
    codebook_layer: int,
    codebook_head_idx: int,
    position: int,
    orig_tokens: Float[Tensor, "batch pos d_model"],
    new_tokens: Float[Tensor, "batch pos d_model"],
    patching_method: Callable,
    orig_cache: Optional[ActivationCache] = None,
    new_cache: Optional[ActivationCache] = None,
) -> Float[Tensor, "batch pos d_vocab"]:
    """
    Extracts the cache from a patched model that used patch_attn_codebook,
    we use this to test that our activation patching is working as expected
    (ie. caches that should be patched over are and caches prior are left the same,
    caches after should not be identical to either orig_cache or new_cache).
    """
    if orig_cache is None:
        _, orig_cache = cb_model.run_with_cache(orig_tokens)
    if new_cache is None:
        _, new_cache = cb_model.run_with_cache(new_tokens)
    activation_name = get_act_name(f"cb_{codebook_head_idx}", codebook_layer, "attn")
    if patching_method == act_patch_attn_codebook:
        hook_fn = partial(patching_method, position=position, new_cache=new_cache)
    elif patching_method == patch_or_freeze_attn_codebook:
        hook_fn = partial(
            patching_method,
            position=position,
            codebook_layer=codebook_layer,
            codebook_head_idx=codebook_head_idx,
            orig_cache=orig_cache,
            new_cache=new_cache,
        )
    elif patching_method == patch_attn_codebook_input:
        hook_fn = partial(
            patching_method,
            position=position,
            reciever_codebook_layer=codebook_layer,
            reciever_codebook_head_idx=codebook_head_idx,
            patched_cache=new_cache,
        )

    cb_model.add_hook(name=activation_name, hook=hook_fn, dir="fwd")
    _, modified_cache = cb_model.run_with_cache(orig_tokens)
    cb_model.reset_hooks()
    return modified_cache


def test_patch_attn_codebook(random_data_setup):
    cb_model, orig_tokens, new_tokens, orig_cache, new_cache = random_data_setup
    codebook_layer_to_change = 4
    codebook_head_to_change = 2
    modified_cache = extract_cache_for_run_using_specified_patching_method(
        cb_model=cb_model,
        codebook_layer=codebook_layer_to_change,
        codebook_head_idx=codebook_head_to_change,
        position=-1,
        orig_tokens=orig_tokens,
        new_tokens=new_tokens,
        patching_method=act_patch_attn_codebook,
    )

    # All caches before the codebook change should be the same
    for layer_idx in range(codebook_layer_to_change):
        for codebook_head_idx in range(cb_model.cfg.n_heads):
            activation_name = get_act_name(f"cb_{codebook_head_idx}", layer_idx, "attn")
            assert t.equal(
                modified_cache[activation_name],
                orig_cache[activation_name],
            )

    # The codebook cache that got changed should equal the new cache for the final position
    activation_name = get_act_name(
        f"cb_{codebook_head_to_change}", codebook_layer_to_change, "attn"
    )
    assert t.equal(
        new_cache[activation_name][:, -1],
        modified_cache[activation_name][:, -1],
    )

    # The codebook cache after the change should not all equal the new cache and not all equal the original_cache
    total_number_caches = 0
    equal_to_orig = 0
    equal_to_new = 0
    for layer_idx in range(codebook_layer_to_change + 1, cb_model.cfg.n_layers):
        for codebook_head_idx in range(cb_model.cfg.n_heads):
            total_number_caches += 1
            activation_name = get_act_name(f"cb_{codebook_head_idx}", layer_idx, "attn")
            if t.equal(
                modified_cache[activation_name],
                orig_cache[activation_name],
            ):
                equal_to_orig += 1
            if t.equal(
                modified_cache[activation_name],
                new_cache[activation_name],
            ):
                equal_to_new += 1

    assert equal_to_new < total_number_caches, print(
        f"equal_to_new: {equal_to_new}, total_number_caches: {total_number_caches}"
    )
    assert equal_to_orig < total_number_caches, print(
        f"equal_to_orig: {equal_to_orig}, total_number_caches: {total_number_caches}"
    )


def test_patch_or_freeze_attn_codebook(random_data_setup):
    cb_model, orig_tokens, new_tokens, orig_cache, new_cache = random_data_setup
    codebook_layer_to_change = 4
    codebook_head_to_change = 2
    position = -1
    modified_cache = extract_cache_for_run_using_specified_patching_method(
        cb_model=cb_model,
        codebook_layer=codebook_layer_to_change,
        codebook_head_idx=codebook_head_to_change,
        position=position,
        orig_tokens=orig_tokens,
        new_tokens=new_tokens,
        patching_method=patch_or_freeze_attn_codebook,
    )

    # All caches beside the codebook to change should remain the same
    for layer_idx in range(codebook_layer_to_change):
        for codebook_head_idx in range(cb_model.cfg.n_heads):
            activation_name = get_act_name(f"cb_{codebook_head_idx}", layer_idx, "attn")
            if activation_name == get_act_name(
                f"cb_{codebook_head_to_change}", codebook_layer_to_change, "attn"
            ):
                continue
            assert t.equal(
                modified_cache[activation_name],
                orig_cache[activation_name],
            )

    # The codebook cache that got changed should equal the new cache for the final position
    activation_name = get_act_name(
        f"cb_{codebook_head_to_change}", codebook_layer_to_change, "attn"
    )
    assert t.equal(
        modified_cache[activation_name][:, position, :],
        new_cache[activation_name][:, position, :],
    )


def test_patch_attn_codebook_input(random_data_setup):
    cb_model, orig_tokens, new_tokens, orig_cache, new_cache = random_data_setup
    codebook_layer_to_change = 4
    codebook_head_to_change = 2
    position = -1
    modified_cache = extract_cache_for_run_using_specified_patching_method(
        cb_model=cb_model,
        codebook_layer=codebook_layer_to_change,
        codebook_head_idx=codebook_head_to_change,
        position=position,
        orig_tokens=orig_tokens,
        new_tokens=new_tokens,
        patching_method=patch_attn_codebook_input,
    )

    # All caches before the codebook change should be the same
    for layer_idx in range(codebook_layer_to_change):
        for codebook_head_idx in range(cb_model.cfg.n_heads):
            activation_name = get_act_name(f"cb_{codebook_head_idx}", layer_idx, "attn")
            assert t.equal(
                modified_cache[activation_name],
                orig_cache[activation_name],
            )

    # The codebook cache that got changed should equal the new cache for the final position
    activation_name = get_act_name(
        f"cb_{codebook_head_to_change}", codebook_layer_to_change, "attn"
    )
    assert t.equal(
        new_cache[activation_name][:, -1],
        modified_cache[activation_name][:, -1],
    )

    # The codebook cache after the change should not all equal the new cache and not all equal the original_cache
    total_number_caches = 0
    equal_to_orig = 0
    equal_to_new = 0
    for layer_idx in range(codebook_layer_to_change + 1, cb_model.cfg.n_layers):
        for codebook_head_idx in range(cb_model.cfg.n_heads):
            total_number_caches += 1
            activation_name = get_act_name(f"cb_{codebook_head_idx}", layer_idx, "attn")
            if t.equal(
                modified_cache[activation_name],
                orig_cache[activation_name],
            ):
                equal_to_orig += 1
            if t.equal(
                modified_cache[activation_name],
                new_cache[activation_name],
            ):
                equal_to_new += 1

    assert equal_to_new < total_number_caches, print(
        f"equal_to_new: {equal_to_new}, total_number_caches: {total_number_caches}"
    )
    assert equal_to_orig < total_number_caches, print(
        f"equal_to_orig: {equal_to_orig}, total_number_caches: {total_number_caches}"
    )


def test_act_vs_path_results(travel_data_setup):
    (
        cb_model,
        orig_tokens,
        new_tokens,
        incorrect_correct_toks,
        orig_cache,
        new_cache,
    ) = travel_data_setup
    sender_codebook_layer = 4
    sender_codebook_head_idx = 2
    reciever_codebook_layer = 8
    reciever_codebook_head_idx = 12
    patch_position = -1
    answer_position = -1

    assert sender_codebook_layer < reciever_codebook_layer

    activation_patch_sender_logits = codebook_activation_patcher(
        cb_model=cb_model,
        codebook_layer=sender_codebook_layer,
        codebook_head_idx=sender_codebook_head_idx,
        position=patch_position,
        orig_tokens=orig_tokens,
        new_tokens=new_tokens,
    )

    activation_patch_sender_performance = compute_average_logit_difference(
        activation_patch_sender_logits, incorrect_correct_toks, answer_position=-1
    )

    activation_patch_reciever = codebook_activation_patcher(
        cb_model=cb_model,
        codebook_layer=reciever_codebook_layer,
        codebook_head_idx=reciever_codebook_head_idx,
        position=patch_position,
        orig_tokens=orig_tokens,
        new_tokens=new_tokens,
    )

    activation_patch_reciever_performance = compute_average_logit_difference(
        activation_patch_reciever,
        incorrect_correct_toks,
        answer_position=answer_position,
    )

    path_patch_sender_to_receiver_logits = basic_codebook_path_patch(
        cb_model=cb_model,
        sender_codebook_layer=sender_codebook_layer,
        sender_codebook_head_idx=sender_codebook_head_idx,
        reciever_codebook_layer=reciever_codebook_layer,
        reciever_codebook_head_idx=reciever_codebook_head_idx,
        patch_position=patch_position,
        orig_tokens=orig_tokens,
        new_tokens=new_tokens,
        orig_cache=orig_cache,
        new_cache=new_cache,
    )

    path_patch_sender_to_receiver_performance = compute_average_logit_difference(
        path_patch_sender_to_receiver_logits,
        incorrect_correct_toks,
        answer_position=answer_position,
    )

    assert not t.equal(
        activation_patch_sender_performance, activation_patch_reciever_performance
    ), f"activation_patch_sender_performance: {activation_patch_sender_performance}, activation_patch_reciever_performance: {activation_patch_reciever_performance}"
    assert not t.equal(
        activation_patch_sender_performance, path_patch_sender_to_receiver_performance
    ), f"activation_patch_sender_performance: {activation_patch_sender_performance}, path_patch_sender_to_receiver_performance: {path_patch_sender_to_receiver_performance}"
    assert not t.equal(
        activation_patch_reciever_performance, path_patch_sender_to_receiver_performance
    ), f"activation_patch_reciever_performance: {activation_patch_reciever_performance}, path_patch_sender_to_receiver_performance: {path_patch_sender_to_receiver_performance}"
