"""Util functions for codebook features."""
import re
import typing
from collections import namedtuple
from functools import partial

import plotly.express as px
import torch
import torch.nn.functional as F
from termcolor import colored
from tqdm import tqdm
from transformer_lens import utils as tl_utils


def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    """Show an image."""
    px.imshow(
        tl_utils.to_numpy(tensor),
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs,
    ).show(renderer)


def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    """Show a line plot."""
    px.line(tl_utils.to_numpy(tensor), labels={"x": xaxis, "y": yaxis}, **kwargs).show(
        renderer
    )


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    """Show a scatter plot."""
    x = tl_utils.to_numpy(x)
    y = tl_utils.to_numpy(y)
    px.scatter(
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    ).show(renderer)


def logits_to_pred(logits, tokenizer, k=5):
    """Convert logits to top-k predictions."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = sorted_logits.softmax(dim=-1)
    topk_preds = [tokenizer.convert_ids_to_tokens(e) for e in sorted_indices[:, -1, :k]]
    topk_preds = [
        tokenizer.convert_tokens_to_string([e]) for batch in topk_preds for e in batch
    ]
    return [(topk_preds[i], probs[:, -1, i].item()) for i in range(len(topk_preds))]


def patch_codebook_ids(
    corrupted_codebook_ids, hook, pos, cache, cache_pos=None, code_idx=None
):
    """Patch codebook ids with cached ids."""
    if cache_pos is None:
        cache_pos = pos
    if code_idx is None:
        corrupted_codebook_ids[:, pos] = cache[hook.name][:, cache_pos]
    else:
        for code_id in range(32):
            if code_id in code_idx:
                corrupted_codebook_ids[:, pos, code_id] = cache[hook.name][
                    :, cache_pos, code_id
                ]
            else:
                corrupted_codebook_ids[:, pos, code_id] = -1

    return corrupted_codebook_ids


def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
    """Calculate the average logit difference between the answer and the other token."""
    # Only the final logits are relevant for the answer
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()


def normalize_patched_logit_diff(
    patched_logit_diff,
    base_average_logit_diff,
    corrupted_average_logit_diff,
):
    """Normalize the patched logit difference."""
    # Subtract corrupted logit diff to measure the improvement,
    # divide by the total improvement from clean to corrupted to normalise
    # 0 means zero change, negative means actively made worse,
    # 1 means totally recovered clean performance, >1 means actively *improved* on clean performance
    return (patched_logit_diff - corrupted_average_logit_diff) / (
        base_average_logit_diff - corrupted_average_logit_diff
    )


def features_to_tokens(cb_key, cb_acts, num_codes, n=10):
    """Returns the set of token ids each codebook feature activates on."""
    codebook_ids = cb_acts[cb_key]
    features_tokens = [[] for _ in range(num_codes)]
    for i in tqdm(range(codebook_ids.shape[0])):
        for j in range(codebook_ids.shape[1]):
            for k in range(codebook_ids.shape[2]):
                features_tokens[codebook_ids[i, j, k]].append((i, j))

    return features_tokens


def color_red(tokens, red_idx, tokenizer, n=3, separate_states=True):
    """Separate states with a dash and color red the tokens in red_idx."""
    ret_string = ""
    itr_over_red_idx = 0
    tokens_enumerate = enumerate(tokens)
    if tokens[0] == tokenizer.bos_token_id:
        next(tokens_enumerate)
        if red_idx[0] == 0:
            itr_over_red_idx += 1
    last_colored_token_dist = 0
    last_token_added = 0
    for i, c in tokens_enumerate:
        if separate_states and i % 2 == 1:
            ret_string += "-"
        if itr_over_red_idx < len(red_idx) and i == red_idx[itr_over_red_idx]:
            c_str = tokenizer.decode(c)
            if last_colored_token_dist > n + 1:  # missed at least one token
                if last_token_added + 1 < i - n:
                    ret_string += " ... "
                ret_string += tokenizer.decode(
                    tokens[max(last_token_added + 1, i - n) : i]
                )
            ret_string += colored(c_str, "red")
            last_token_added = i
            itr_over_red_idx += 1
            last_colored_token_dist = 0
        elif last_colored_token_dist <= n:
            ret_string += tokenizer.decode(c)
            last_token_added = i
        last_colored_token_dist += 1
    return ret_string


def tkn_print(ll, tokens, tokenizer, separate_states, n=3, max_examples=20):
    """Formats and prints the tokens in ll."""
    # indices = np.random.choice(len(ll), min(len(ll), max_examples), replace=False)
    indices = range(len(ll))
    print_output = ""
    current_example = 0
    total_examples = 0
    tokens_to_color_red = []
    for idx in indices:
        if total_examples > max_examples:
            break
        i, j = ll[idx]

        if i != current_example:
            if current_example != 0:
                print_output += (
                    f"{current_example}: "
                    + color_red(
                        tokens[current_example],
                        tokens_to_color_red,
                        tokenizer,
                        n=n,
                        separate_states=separate_states,
                    )
                    + "\n"
                )
                total_examples += 1
            current_example = i
            tokens_to_color_red = []
        tokens_to_color_red.append(j)

    print_output += colored("********************************************", "green")
    return print_output


def print_ft_tkns(
    ft_tkns,
    tokens,
    tokenizer,
    separate_states=False,
    n=3,
    start=0,
    stop=1000,
    indices=None,
    max_examples=200,
):
    """Prints the tokens for the codebook features."""
    indices = list(range(start, stop)) if indices is None else indices
    num_tokens = len(tokens) * len(tokens[0])
    token_act_freqs = []
    token_acts = []
    for i in indices:
        tkns = ft_tkns[i]
        token_act_freqs.append(100 * len(tkns) / num_tokens)
        if len(tkns) > 0:
            tkn_acts = tkn_print(
                tkns, tokens, tokenizer, separate_states, n=n, max_examples=max_examples
            )
            token_acts.append(tkn_acts)
        else:
            token_acts.append("")
    return token_act_freqs, token_acts


def patch_in_codes(run_cb_ids, hook, pos, code, code_pos=None):
    """Patch in the `code` at `run_cb_ids`."""
    pos = slice(None) if pos is None else pos
    code_pos = slice(None) if code_pos is None else code_pos

    if code_pos == "append":
        assert pos == slice(None)
        run_cb_ids = F.pad(run_cb_ids, (0, 1), mode="constant", value=code)
    if isinstance(pos, typing.Iterable) or isinstance(pos, typing.Iterable):
        for p in pos:
            run_cb_ids[:, p, code_pos] = code
    else:
        run_cb_ids[:, pos, code_pos] = code
    return run_cb_ids


def get_cb_layer_name(cb_at, layer_idx, head_idx=None):
    """Get the layer name used to store hooks/cache."""
    if head_idx is None:
        return f"blocks.{layer_idx}.{cb_at}.codebook_layer.hook_codebook_ids"
    else:
        return f"blocks.{layer_idx}.{cb_at}.codebook_layer.codebook.{head_idx}.hook_codebook_ids"


def get_cb_layer_names(layer, patch_types, n_heads):
    """Get the layer names used to store hooks/cache."""
    layer_names = []
    attn_added, mlp_added = False, False
    if "attn_out" in patch_types:
        attn_added = True
        for head in range(n_heads):
            layer_names.append(
                f"blocks.{layer}.attn.codebook_layer.codebook.{head}.hook_codebook_ids"
            )
    if "mlp_out" in patch_types:
        mlp_added = True
        layer_names.append(f"blocks.{layer}.mlp.codebook_layer.hook_codebook_ids")

    for patch_type in patch_types:
        # match patch_type of the pattern attn_\d_head_\d
        attn_head = re.match(r"attn_(\d)_head_(\d)", patch_type)
        if (not attn_added) and attn_head and attn_head[1] == str(layer):
            layer_names.append(
                f"blocks.{layer}.attn.codebook_layer.codebook.{attn_head[2]}.hook_codebook_ids"
            )
        mlp = re.match(r"mlp_(\d)", patch_type)
        if (not mlp_added) and mlp and mlp[1] == str(layer):
            layer_names.append(f"blocks.{layer}.mlp.codebook_layer.hook_codebook_ids")

    return layer_names


def cb_layer_name_to_info(layer_name):
    """Get the layer info from the layer name."""
    layer_name_split = layer_name.split(".")
    layer_idx = int(layer_name_split[1])
    cb_at = layer_name_split[2]
    if cb_at == "mlp":
        head_idx = None
    else:
        head_idx = int(layer_name_split[5])
    return cb_at, layer_idx, head_idx


def get_hooks(code, cb_at, layer_idx, head_idx=None, pos=None):
    """Get the hooks for the codebook features."""
    hook_fns = [
        partial(patch_in_codes, pos=pos, code=code[i]) for i in range(len(code))
    ]
    return [
        (get_cb_layer_name(cb_at[i], layer_idx[i], head_idx[i]), hook_fns[i])
        for i in range(len(code))
    ]


def run_with_codes(
    input, cb_model, code, cb_at, layer_idx, head_idx=None, pos=None, prepend_bos=True
):
    """Run the model with the codebook features patched in."""
    hook_fns = [
        partial(patch_in_codes, pos=pos, code=code[i]) for i in range(len(code))
    ]
    cb_model.reset_codebook_metrics()
    cb_model.reset_hook_kwargs()
    fwd_hooks = [
        (get_cb_layer_name(cb_at[i], layer_idx[i], head_idx[i]), hook_fns[i])
        for i in range(len(cb_at))
    ]
    with cb_model.hooks(fwd_hooks, [], True, False) as hooked_model:
        patched_logits, patched_cache = hooked_model.run_with_cache(
            input, prepend_bos=prepend_bos
        )
    return patched_logits, patched_cache


CodeInfoTuple = namedtuple(
    "CodeInfoTuple", ["code", "cb_at", "layer", "head", "pos", "code_pos"]
)


def in_hook_list(list_of_arg_tuples, layer, head=None):
    """Check if the component specified by `layer` and `head` is in the `list_of_arg_tuples`."""
    # if head is not provided, then checks in MLP
    for arg_tuple in list_of_arg_tuples:
        if head is None:
            if arg_tuple.cb_at == "mlp" and arg_tuple.layer == layer:
                return True
        else:
            if (
                arg_tuple.cb_at == "attn"
                and arg_tuple.layer == layer
                and arg_tuple.head == head
            ):
                return True
    return False


# def generate_with_codes(input, code, cb_at, layer_idx, head_idx=None, pos=None, disable_other_comps=False):
def generate_with_codes(
    input,
    cb_model,
    list_of_code_infos=(),
    disable_other_comps=False,
    automata=None,
    generate_kwargs=None,
):
    """Model's generation with the codebook features patched in."""
    if generate_kwargs is None:
        generate_kwargs = {}
    hook_fns = [
        partial(patch_in_codes, pos=tupl.pos, code=tupl.code)
        for tupl in list_of_code_infos
    ]
    fwd_hooks = [
        (get_cb_layer_name(tupl.cb_at, tupl.layer, tupl.head), hook_fns[i])
        for i, tupl in enumerate(list_of_code_infos)
    ]
    cb_model.reset_hook_kwargs()
    if disable_other_comps:
        for layer, cb in cb_model.all_codebooks.items():
            for head_idx, head in enumerate(cb[0].codebook):
                if not in_hook_list(list_of_code_infos, layer, head_idx):
                    head.set_hook_kwargs(
                        disable_topk=1, disable_for_tkns=[-1], keep_k_codes=False
                    )
            if not in_hook_list(list_of_code_infos, layer):
                cb[1].set_hook_kwargs(
                    disable_topk=1, disable_for_tkns=[-1], keep_k_codes=False
                )
    with cb_model.hooks(fwd_hooks, [], True, False) as hooked_model:
        gen = hooked_model.generate(input, **generate_kwargs)
        print(gen)
    return automata.seq_to_traj(gen)[0] if automata is not None else gen


def kl_div(logits1, logits2, pos=-1, reduction="batchmean"):
    """Calculate the KL divergence between the logits at `pos`."""
    logits1_last, logits2_last = logits1[:, pos, :], logits2[:, pos, :]
    # calculate kl divergence between clean and mod logits last
    return F.kl_div(
        F.log_softmax(logits1_last, dim=-1),
        F.log_softmax(logits2_last, dim=-1),
        log_target=True,
        reduction=reduction,
    )


def JSD(logits1, logits2, pos=-1, reduction="batchmean"):
    """Compute the Jensen-Shannon divergence between two distributions."""
    if len(logits1.shape) == 3:
        logits1, logits2 = logits1[:, pos, :], logits2[:, pos, :]

    probs1 = F.softmax(logits1, dim=-1)
    probs2 = F.softmax(logits2, dim=-1)

    total_m = (0.5 * (probs1 + probs2)).log()

    loss = 0.0
    loss += F.kl_div(
        total_m,
        F.log_softmax(logits1, dim=-1),
        log_target=True,
        reduction=reduction,
    )
    loss += F.kl_div(
        total_m,
        F.log_softmax(logits2, dim=-1),
        log_target=True,
        reduction=reduction,
    )
    return 0.5 * loss


def residual_stream_patching_hook(resid_pre, hook, cache, position: int):
    """Patch in the codebook features at `position` from `cache`."""
    clean_resid_pre = cache[hook.name]
    resid_pre[:, position, :] = clean_resid_pre[:, position, :]
    return resid_pre


def find_code_changes(cache1, cache2, pos=None):
    """Find the codebook codes that are different between the two caches."""
    for k in cache1.keys():
        if "codebook" in k:
            c1 = cache1[k][0, pos]
            c2 = cache2[k][0, pos]
            if not torch.all(c1 == c2):
                print(cb_layer_name_to_info(k), c1.tolist(), c2.tolist())
                print(cb_layer_name_to_info(k), c1.tolist(), c2.tolist())


def common_codes_in_cache(cache_codes, threshold=0.0):
    """Returns the common code in the cache."""
    codes, counts = torch.unique(cache_codes, return_counts=True, sorted=True)
    counts = counts.float() * 100
    counts /= cache_codes.shape[1]
    counts, indices = torch.sort(counts, descending=True)
    codes = codes[indices]
    indices = counts > threshold
    codes, counts = codes[indices], counts[indices]
    return codes, counts
