from jaxtyping import Float
from torch import Tensor
from data import TravelToCityDataset
import torch as t
from codebook_features import models

def compute_average_logit_difference(logits: Float[Tensor, "batch pos d_vocab"],
                                     incorrect_correct_responses: Float[Tensor, "batch 2"],
                                     answer_position: int = -1) -> Float[Tensor, ""]:
    """
    Compute the average logit difference between incorrect and correct tokens across the batch.
    
    Parameters:
    logits (Float[Tensor, "batch pos d_vocab"]): Logits tensor
    incorrect_correct_responses (Float[Tensor, "batch 2"]): Tensor of incorrect and correct responses
    answer_position (int, optional): Position of the answer. Defaults to -1.

    Returns:
    Float[Tensor, ""]: Average logit difference
    """
    # Get the logits at the answer position
    answer_pos_logits = logits[:, answer_position, :] 

    # Gather the correct and incorrect logits
    correct_incorrect_logits = answer_pos_logits.gather(dim=-1, index=incorrect_correct_responses)

    # Separate the correct and incorrect logits
    correct_logits, incorrect_logits = correct_incorrect_logits[:, 0], correct_incorrect_logits[:, 1]

    # Compute and return the average logit difference
    return (correct_logits - incorrect_logits).mean()

if __name__ == "__main__":
    DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
    model_name_or_path = "EleutherAI/pythia-410m-deduped"
    pre_trained_path = "taufeeque/pythia410m_wikitext_attn_codebook_model"
    orig_cb_model = models.wrap_codebook(
        model_or_path=model_name_or_path, pretrained_path=pre_trained_path
    )
    orig_cb_model = orig_cb_model.to(DEVICE).eval()

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
    dataset = TravelToCityDataset(n_examples=100)
    correct_prompts = dataset.clean_prompts
    logits = cb_model(correct_prompts)
    incorrect_correct_responses = dataset.correct_incorrects
    incorrect_correct_toks = t.concat([cb_model.to_tokens(response, prepend_bos=False).T for response in incorrect_correct_responses])
    avg_logits = compute_average_logit_difference(logits, incorrect_correct_toks)