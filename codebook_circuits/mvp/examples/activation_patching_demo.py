import plotly.graph_objects as go
import torch as t
from codebook_circuits.mvp.codebook_act_patching import (
    codebook_activation_patcher,
    iter_codebook_act_patching,
)
from codebook_circuits.mvp.data import TravelToCityDataset
from codebook_circuits.mvp.utils import (
    compute_average_logit_difference,
    setup_pythia410M_hooked_model,
)

# Set up model
cb_model = setup_pythia410M_hooked_model()

## Creating some data for testing and development
dataset = TravelToCityDataset(n_examples=50)
responses = dataset.correct_incorrects
incorrect_correct_toks = t.cat(
    [cb_model.to_tokens(response, prepend_bos=False).T for response in responses]
)
orig_input = dataset.clean_prompts
new_input = dataset.corrupted_prompts

# Benchmarking Original Model Performance
orig_logits = cb_model(orig_input)
original_perfomance = compute_average_logit_difference(
    orig_logits, incorrect_correct_toks, answer_position=-1
)
orig_logits = cb_model(orig_input)
original_perfomance = compute_average_logit_difference(
    orig_logits, incorrect_correct_toks, answer_position=-1
)
print(f"Original Model Perfomance: {original_perfomance}")
patched_logits = codebook_activation_patcher(
    cb_model=cb_model,
    codebook_layer=6,
    codebook_head_idx=3,
    position=-1,
    orig_tokens=orig_input,
    new_tokens=new_input,
)
print(
    f"Patched Model Perfomance: {compute_average_logit_difference(patched_logits, incorrect_correct_toks, answer_position=-1)}"
)

# Perform iterative codebook activation patching
orig_tokens = cb_model.to_tokens(orig_input, prepend_bos=False)
_, new_cache = cb_model.run_with_cache(new_input)
cb_patching_array = iter_codebook_act_patching(
    cb_model=cb_model,
    orig_tokens=orig_tokens,
    new_cache=new_cache,
    incorrect_correct_toks=incorrect_correct_toks,
    response_position=-1,
    patch_position=-1,
)

cb_patching_array_np = cb_patching_array.numpy()
heatmap = go.Heatmap(z=cb_patching_array_np, colorscale="RdBu")
fig = go.Figure(data=[heatmap])
fig.update_xaxes(title_text="Codebook Head Index")
fig.update_yaxes(title_text="Model Layer")
fig.update_layout(title_text="Activation Patching CodeBooks")
fig.show()
