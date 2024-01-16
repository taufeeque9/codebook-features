import torch as t
from codebook_circuits.mvp.data import TravelToCityDataset
from codebook_features import models
from codebook_features.models import HookedTransformerCodebookModel
from jaxtyping import Float
from torch import Tensor


def compute_average_logit_difference(
    logits: Float[Tensor, "batch pos d_vocab"],
    incorrect_correct_toks: Float[Tensor, "batch 2"],
    answer_position: int = -1,
) -> Float[Tensor, ""]:
    """
    Compute the average logit difference between incorrect and correct tokens across the batch.

    Parameters:
    logits (Float[Tensor, "batch pos d_vocab"]): Logits tensor
    incorrect_correct_responses (Float[Tensor, "batch 2"]): Tensor of incorrect and correct responses
    answer_position (int, optional): Position of the answer. Defaults to -1.

    Returns:
    Float[Tensor, ""]: Average logit difference
    """
    # Do not require gradients
    logits = logits.detach()

    # Get the logits at the answer position
    answer_pos_logits = logits[:, answer_position, :]

    # Gather the correct and incorrect logits
    correct_incorrect_logits = answer_pos_logits.gather(
        dim=-1, index=incorrect_correct_toks
    )

    # Separate the correct and incorrect logits
    correct_logits, incorrect_logits = (
        correct_incorrect_logits[:, 0],
        correct_incorrect_logits[:, 1],
    )

    # Compute and return the average logit difference
    return (correct_logits - incorrect_logits).mean()


def logit_change_metric(
    new_logits: Float[Tensor, "batch pos d_vocab"],
    orig_logits: Float[Tensor, "batch pos d_vocab"],
    incorrect_correct_toks: Float[Tensor, "batch 2"],
    answer_position: int = -1,
) -> Float[Tensor, ""]:
    """
    Compute the logit change metric between two sets of logits.

    Parameters:
    new_logits (Float[Tensor, "batch pos d_vocab"]): New logits
    orig_logits (Float[Tensor, "batch pos d_vocab"]): Original logits
    incorrect_correct_toks (Float[Tensor, "batch 2"]): Tensor of incorrect and correct tokens
    answer_position (int, optional): Position of the answer. Defaults to -1.

    Returns:
    Float[Tensor, ""]: Logit change metric
    """
    orig_avg_logit_diff = compute_average_logit_difference(
        orig_logits, incorrect_correct_toks, answer_position
    )
    new_avg_logit_diff = compute_average_logit_difference(
        new_logits, incorrect_correct_toks, answer_position
    )

    return (new_avg_logit_diff - orig_avg_logit_diff) / orig_avg_logit_diff


def setup_pythia410M_hooked_model() -> HookedTransformerCodebookModel:
    # Set device variable
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    ## Loading and setting up hooked codebook model
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
