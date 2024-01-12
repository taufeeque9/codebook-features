import torch as t
from codebook_features import models
from data import TravelToCityDataset
from utils import compute_average_logit_difference
from cb_activation_patching import codebook_activation_patcher

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

## Creating some data for testing and development
dataset = TravelToCityDataset(n_examples=10)
responses = dataset.correct_incorrects
answer_tokens = t.cat([cb_model.to_tokens(response, prepend_bos=False).T for response in responses])
orig_input = dataset.clean_prompts
new_input = dataset.corrupted_prompts
orig_logits = cb_model(orig_input)
original_perfomance = compute_average_logit_difference(orig_logits, answer_tokens, answer_position=-1)
print(f'Original Model Perfomance: {original_perfomance}')
patched_logits = codebook_activation_patcher(cb_model = cb_model, 
                                                codebook_layer = 6, 
                                                codebook_head_idx = 3, 
                                                position = -1, 
                                                orig_tokens = orig_input, 
                                                new_tokens = new_input)
print(f'Patched Model Perfomance: {compute_average_logit_difference(patched_logits, answer_tokens, answer_position=-1)}')
