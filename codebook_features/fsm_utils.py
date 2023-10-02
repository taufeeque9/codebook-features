"""Util functions for the toy model."""
import itertools
from functools import partial

import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from functorch.dim import dims
from tqdm import tqdm

from codebook_features import utils


def valid_input(input: str, fsm):
    """Check if the input is a valid bigram/trigram string that can be generated by the fsm."""
    assert len(input) <= fsm.digits + 1
    if len(input) < fsm.digits + 1:
        return True
    state = fsm.seq_to_traj(input)[0][0]
    ns_start_token = input[-1]
    possible_next_states = fsm.get_out_neighbors(state)
    possible_start_tokens = [fsm.token_repr(s)[0] for s in possible_next_states]
    return ns_start_token in possible_start_tokens


def partition_input_on_codebook(
    cb_model,
    fsm,
    cb_at,
    layer,
    gcb_idx,
    input_len=2,
):
    """Partition the input space based on the codes at the specified component.

    Args:
        cb_model: hooked codebook model.
        fsm: finite state machine.
        cb_at: the component codebook is applied at. Can be "attn" or "mlp".
        layer: the layer index of the component.
        gcb_idx: if the component is a group codebook, the index of the codebook.
        input_len: length of the input. This specifies the input space as all possible
            valid inputs of length `input_len` that can be generated by `fsm`.

    Returns:
        A dictionary mapping code indices to the list of inputs that map to that code.
    """
    cache_str = utils.get_cb_hook_key(cb_at, layer, gcb_idx)
    partition = {}
    chars = [str(c) for c in range(fsm.representation_base)]
    input_range = itertools.product(chars, repeat=input_len)
    for inp_tuple in tqdm(input_range):
        inp = "".join(inp_tuple)
        if not valid_input(inp, fsm):
            continue
        mod_input = cb_model.to_tokens(inp, prepend_bos=True).to("cuda")
        mod_logits, mod_cache = cb_model.run_with_cache(mod_input)
        mod_indices = mod_cache[cache_str][0, -1].tolist()
        for mod_index in mod_indices:
            if mod_index not in partition:
                partition[mod_index] = []
            partition[mod_index].append(inp)
    return partition


def get_next_state_probs(state, model, fsm, fwd_hooks=None, prepend_bos=True):
    """Get the top next state probabilities given by the model.

    Args:
        state: a single state or a tensor of states in the fsm.
        model: the hooked codebook model.
        fsm: the finite state machine.
        fwd_hooks: hooks to apply when running the `model`.
        prepend_bos: whether to prepend the bos token to the state.

    Returns:
        next_state_preds: the top `edges` next state predictions for each state.
        next_state_probs: the top `edges` next state probabilities for each state.
    """
    if isinstance(state, int):
        state_str = fsm.traj_to_str([state])
        state = model.to_tokens(state_str, prepend_bos=prepend_bos).to("cuda")
    elif not isinstance(state, torch.Tensor):
        raise ValueError("state must be an int or a tensor of state inputs")

    if fwd_hooks is not None:

        def model_run(x):
            return model.run_with_hooks(x, fwd_hooks=fwd_hooks)

    else:

        def model_run(x):
            return model(x)

    next_state_probs = torch.zeros((state.shape[0], fsm.N)).to("cuda")
    base = fsm.representation_base
    for next_token in range(base):
        next_token_input = F.pad(state, (0, 1), value=next_token).to("cuda")
        next_token_logits = model_run(next_token_input)
        if isinstance(next_token_logits, dict):
            next_token_logits = next_token_logits["logits"]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        next_state_prob = (
            next_token_probs[:, -2, next_token].unsqueeze(-1)
            * next_token_probs[:, -1, :base]
        )
        next_state_probs[
            :, next_token * base : (next_token + 1) * base
        ] = next_state_prob

    # filter next_state_probs to only include the top `edges`
    next_state_probs, next_state_preds = torch.topk(
        next_state_probs, fsm.edges, dim=-1, sorted=True
    )
    return next_state_preds, next_state_probs


def correct_next_state_probs(state, next_state_probs, fsm, print_info=""):
    """Get the accuracy of the next state predictions for the given state.

    Args:
        state: a single state or a tensor of states in the fsm.
        next_state_probs: the next state probabilities for each state.
        fsm: the finite state machine.
        print_info: whether to print incorrect and correct transitions.
            Can be "i" for incorrect, "c" for correct, or "ic" for both.
            Defaults to "".

    Returns:
        the accuracy of the next state predictions for the given state(s).
    """
    if isinstance(next_state_probs, tuple):
        next_states = next_state_probs[0]
    elif isinstance(next_state_probs, torch.Tensor):
        next_states = next_state_probs
    else:
        raise ValueError("next_state_probs must be a tensor or tuple")

    if isinstance(state, list):
        state = [int(s) for s in state]
    elif isinstance(state, int):
        state = [state] * next_states.shape[0]
    else:
        raise ValueError("state must be an int or a list of ints.")

    next_states_pred = torch.zeros((next_states.shape[0], fsm.N), dtype=bool).to("cuda")
    next_states_pred.scatter_(1, next_states, True)

    actual_next_states = fsm.transition_matrix[state, :] > 0
    common = actual_next_states * next_states_pred.cpu().numpy()
    accuracy = common.sum(axis=-1) / fsm.edges
    if "i" in print_info:
        incorrect_transitions = next_states_pred - common
        for i, s in enumerate(state):
            print(
                f"incorrect transitions: {s} ->  {incorrect_transitions[i].nonzero().tolist()}"
            )
    if "c" in print_info:
        print(f"Correct transitions: {state} ->  {common}")
    return accuracy


def first_transition_accuracy(model, fsm, fwd_hooks=None, prepend_bos=True):
    """Get the average accuracy of the first transition given by the model for all states."""
    avg_acc = 0
    for state in tqdm(range(fsm.N)):
        nsp = get_next_state_probs(state, model, fsm, fwd_hooks, prepend_bos)[0]
        acc = correct_next_state_probs(state, nsp, fsm)[0]
        avg_acc += acc
    avg_acc /= fsm.N
    return avg_acc


def plot_js_div(
    code_groups_for_all_comps,
    layer,
    cb_at,
    gcb_idx,
    js_divs_state_pairs,
    fig=None,
    row=None,
    col=None,
    **fig_kwargs,
):
    """Plot the histogram of JSD between random pairs of states and states grouped by codes."""
    group_js_divs = {}
    code_groups = code_groups_for_all_comps[(layer, cb_at, gcb_idx)]
    for code, grouped_states in code_groups.items():
        if len(grouped_states) < 2:
            continue
        group_js_divs[code] = []
        for ia, sa in enumerate(grouped_states):
            for sb in grouped_states[ia + 1 :]:
                group_js_divs[code].append(js_divs_state_pairs[(sa, sb)])

    avg_group_js_divs = {
        code: sum(grp_js_divs) / len(grp_js_divs)
        for code, grp_js_divs in group_js_divs.items()
    }

    js_values = list(js_divs_state_pairs.values())
    input_len = len(list(js_divs_state_pairs.keys())[0][0])
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=js_values,
            nbinsx=100,
            histnorm="probability",
            name=f"{'bigram' if input_len == 2 else 'trigram'} input pairs",
            legendgroup="h1",
            marker=go.histogram.Marker(color="blue"),
            **fig_kwargs,
        ),
        row=row,
        col=col,
    )
    avg_group_js_divs_values = list(avg_group_js_divs.values())
    fig.add_trace(
        go.Histogram(
            x=avg_group_js_divs_values,
            nbinsx=100,
            histnorm="probability",
            name=f"{'bigram' if input_len == 2 else 'trigram'} input code groups",
            legendgroup="h2",
            marker=go.histogram.Marker(color="red"),
            **fig_kwargs,
        ),
        row=row,
        col=col,
    )
    return fig


def patch_in_codes_var_pos(run_cb_ids, hook, pos, code, code_pos=None):
    """Patch in the `code` at `run_cb_ids` at positions specified by `pos`."""
    a, b, c = dims(sizes=run_cb_ids.shape)
    code_prime = code.unsqueeze(0).repeat(a.size, 1)
    run_cb_ids[a, pos[a], c] = code_prime[a, c]
    return run_cb_ids


def run_with_codes_var_pos(
    input, cb_model, code, cb_at, layer_idx, head_idx=None, pos=None, prepend_bos=True
):
    """Run the model with the codebook features patched in."""
    hook_fns = [
        partial(patch_in_codes_var_pos, pos=pos, code=code[i]) for i in range(len(code))
    ]
    cb_model.reset_codebook_metrics()
    cb_model.reset_hook_kwargs()
    fwd_hooks = [
        (utils.get_cb_hook_key(cb_at[i], layer_idx[i], head_idx[i]), hook_fns[i])
        for i in range(len(cb_at))
    ]
    with cb_model.hooks(fwd_hooks, [], True, False) as hooked_model:
        patched_logits, patched_cache = hooked_model.run_with_cache(
            input, prepend_bos=prepend_bos
        )
    return patched_logits, patched_cache


def get_layers_from_patching_str(patching):
    """Get the layers from the patching string."""
    layers = patching.split("_")[0].split(",")
    layers = [int(layer[1:]) for layer in layers]
    return layers


def clean_patching_name(patching):
    """Clean the patching name."""
    if patching == "none":
        return "None"
    layers = patching.split("_")[0].split(",")
    cb_at = patching.split("_")[1:]
    cb_at_map = {"attn": "Attn", "mlp": "MLP"}
    layers_map = {"l": "L", "aLL": "All"}
    clean_layers = []
    for layer in layers:
        clean_layers.append(layer)
        for k, v in layers_map.items():
            clean_layers[-1] = clean_layers[-1].replace(k, v)
    clean_patching = ", ".join(clean_layers)
    clean_cb_at = []
    for i_cb_at in cb_at:
        clean_cb_at.append(i_cb_at)
        for k, v in cb_at_map.items():
            clean_cb_at[-1] = clean_cb_at[-1].replace(k, v)
    clean_patching += " " + ", ".join(clean_cb_at)
    return clean_patching
