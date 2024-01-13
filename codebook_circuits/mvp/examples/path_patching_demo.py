import plotly.graph_objects as go
import torch as t
from codebook_circuits.mvp.codebook_patching import basic_codebook_path_patch
from codebook_circuits.mvp.data import TravelToCityDataset
from codebook_circuits.mvp.utils import compute_average_logit_difference
from codebook_features import models

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
dataset = TravelToCityDataset(n_examples=50)
responses = dataset.correct_incorrects
incorrect_correct_toks = t.cat(
    [cb_model.to_tokens(response, prepend_bos=False).T for response in responses]
)
orig_input = dataset.clean_prompts
new_input = dataset.corrupted_prompts
orig_tokens = cb_model.to_tokens(orig_input, prepend_bos=False)
new_tokens = cb_model.to_tokens(new_input, prepend_bos=False)
orig_logits = cb_model(orig_tokens)
original_perfomance = compute_average_logit_difference(
    orig_logits, incorrect_correct_toks, answer_position=-1
)
print(f"Original Model Perfomance: {original_perfomance}")

_, orig_cache = cb_model.run_with_cache(orig_tokens)
_, new_cache = cb_model.run_with_cache(new_tokens)

patched_logits = basic_codebook_path_patch(
    cb_model=cb_model,
    sender_codebook_layer=17,
    sender_codebook_head_idx=5,
    reciever_codebook_layer=19,
    reciever_codebook_head_idx=10,
    patch_position=-1,
    orig_tokens=orig_tokens,
    new_tokens=new_input,
    orig_cache=orig_cache,
    new_cache=new_cache,
)

patched_performance = compute_average_logit_difference(
    patched_logits, incorrect_correct_toks, answer_position=-1
)

print(f"Patched Model Perfomance: {patched_performance}")
