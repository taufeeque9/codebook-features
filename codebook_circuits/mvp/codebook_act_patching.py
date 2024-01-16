# Imports
from functools import partial
from typing import Optional

import torch as t
from codebook_circuits.mvp.more_tl_mods import get_act_name
from codebook_circuits.mvp.utils import logit_change_metric
from codebook_features.models import HookedTransformerCodebookModel
from jaxtyping import Float, Int
from torch import Tensor, zeros
from tqdm import tqdm
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint

## Before jumping into path patching, let's implement the conceptually simpler
## activation patching. This will allow us to conduct sanity checks on the results
## of path patching (as well as be useful in its own right).


## Activation Patching Implementation
def act_patch_attn_codebook(
    orig_codebook: Float[Tensor, "batch pos codes"],
    position: int,
    hook: HookPoint,
    new_cache: ActivationCache,
) -> Float[Tensor, "batch pos codes"]:
    """
    Patch over the codebook at a specific layer and head index with a new codebook (ie. a codebook associated with a corrupted prompt run)

    Args:
        orig_codebook (Float[Tensor, "batch pos codes"]): Original Codebook
        orig_codebook_layer (int): Original Codebook Layer
        orig_codebook_head_idx (int): Original Codebook Head Index
        positions (int): Sequence position to patch over
        hook (HookPoint): TransformerLens Hook Point
        new_cache (ActivationCache): New Activation Cache (associated with a corrupted prompt run for example)

    Return:
        Float[Tensor, "batch pos codes"]: New Codebook
    """
    orig_codebook[:, position, :] = new_cache[hook.name][:, position, :]
    return orig_codebook


def codebook_activation_patcher(
    cb_model: HookedTransformerCodebookModel,
    codebook_layer: int,
    codebook_head_idx: int,
    position: int,
    orig_tokens: Float[Tensor, "batch pos d_model"],
    new_tokens: Optional[Float[Tensor, "batch pos d_model"]] = None,
    new_cache: Optional[ActivationCache] = None,
) -> Float[Tensor, "batch pos d_vocab"]:
    """
    Args:
        cb_model (HookedTransformerCodebookModel): Codebook Model with TL functionality
        codebook_layer (int): Codebook Layer for Codebook to Patch Over
        codebook_id (int): Codebook Id for Codebook to Patch Over
        position (int): Position to Patch Over
        orig_tokens (Tensor): Original tokens (ie. clean tokens), shape (batch, pos, d_model)
        new_tokens (Tensor): New tokens (ie. corrupt tokens), shape (batch, pos, d_model)

    Returns:
        Tensor: Logits, shape (batch, pos, d_vocab)
    """
    if new_cache is None:
        _, new_cache = cb_model.run_with_cache(new_tokens)
    activation_name = get_act_name(f"cb_{codebook_head_idx}", codebook_layer, "attn")
    hook_fn = partial(act_patch_attn_codebook, position=position, new_cache=new_cache)
    patched_logits = cb_model.run_with_hooks(
        orig_tokens, fwd_hooks=[(activation_name, hook_fn)], return_type="logits"
    )
    cb_model.reset_hooks()
    return patched_logits


def iter_codebook_act_patching(
    cb_model: HookedTransformerCodebookModel,
    orig_tokens: Float[Tensor, "batch pos d_model"],
    new_cache: ActivationCache,
    incorrect_correct_toks: Int[Tensor, "batch 2"],
    response_position: int,
    patch_position: int,
) -> Float[Tensor, "layer cb_head_idx"]:
    """
    Iteratively activation patch over each codebook for a given position in each layer and head index.

    Args:
        orig_tokens (Tensor): Original tokens (ie. clean tokens), shape (batch, pos, d_model)
        new_cache (ActivationCache): New Activation Cache (associated with a corrupted prompt run for example)
        position (int): Position to Patch Over

    Returns:
        Float[Tensor, "layer cb_head_idx"]: A performance metric for each layer and head index
    """
    orig_logits = cb_model(orig_tokens)
    n_layers = cb_model.cfg.n_layers
    n_heads = cb_model.cfg.n_heads
    codebook_patch_array = zeros(
        n_layers,
        n_heads,
        dtype=t.float32,
        device=cb_model.cfg.device,
        requires_grad=False,
    )
    for cb_layer in tqdm(range(n_layers), desc="Iterating over Model Layers"):
        for cb_head_idx in range(n_heads):
            patched_logits = codebook_activation_patcher(
                cb_model=cb_model,
                codebook_layer=cb_layer,
                codebook_head_idx=cb_head_idx,
                position=patch_position,
                orig_tokens=orig_tokens,
                new_cache=new_cache,
            )

            perf_metric = logit_change_metric(
                orig_logits=orig_logits,
                new_logits=patched_logits,
                incorrect_correct_toks=incorrect_correct_toks,
                answer_position=response_position,
            )

            codebook_patch_array[cb_layer, cb_head_idx] = perf_metric

    return codebook_patch_array
