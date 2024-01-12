# Imports
from jaxtyping import Float
import torch as t
from codebook_features import models
from torch import Tensor
from data import TravelToCityDataset
from codebook_features.models import HookedTransformerCodebookModel
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint
from more_tl_mods import get_act_name
from functools import partial
from typing import Optional

## Before jumping into path patching, let's implement the conceptually simpler
## activation patching. This will allow us to conduct sanity checks on the results
## of path patching (as well as be useful in its own right).

def patch_attn_codebook(
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
) -> Float[Tensor, "batch pos d_vocab"]:
    """
    Args:
        cb_model (HookedTransformerCodebookModel): Codebook Model with TL functionality
        codebook_layer (int): Codebook Layer for Codebook to Patch Over
        codebook_id (int): Codebook Id for Codebook to Patch Over
        position (int): Position to Patch Over
        orig_tokens (Tensor): Original tokens, shape (batch, pos, d_model)
        new_tokens (Tensor): New tokens, shape (batch, pos, d_model)

    Returns:
        Tensor: Logits, shape (batch, pos, d_vocab)
    """

    _, new_cache = cb_model.run_with_cache(new_tokens)
    activation_name = get_act_name(f'cb_{codebook_head_idx}', codebook_layer, 'attn')
    hook_fn = partial(patch_attn_codebook, position = position, new_cache = new_cache)
    patched_logits = cb_model.run_with_hooks(orig_tokens,
                                             fwd_hooks = [(activation_name, hook_fn)],
                                             return_type = "logits")
    cb_model.reset_hooks()
    return patched_logits

def iterative_cb_act_patching():
    pass

    
