import torch as t
from codebook_circuits.mvp.data import TravelToCityDataset
from codebook_features import models
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
    orig_logits: Float[Tensor, "batch pos d_vocab"],
    new_logits: Float[Tensor, "batch pos d_vocab"],
    incorrect_correct_toks: Float[Tensor, "batch 2"],
    answer_position: int = -1,
) -> Float[Tensor, ""]:
    orig_avg_logit_diff = compute_average_logit_difference(
        orig_logits, incorrect_correct_toks, answer_position
    )
    new_avg_logit_diff = compute_average_logit_difference(
        new_logits, incorrect_correct_toks, answer_position
    )

    return (new_avg_logit_diff - orig_avg_logit_diff) / orig_avg_logit_diff
