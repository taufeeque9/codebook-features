"""Util functions for codebook features."""
import re
import typing
from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from termcolor import colored
from tqdm import tqdm


@dataclass
class CodeInfo:
    """Dataclass for codebook info."""

    code: int
    layer: int
    head: Optional[int]
    cb_at: Optional[str] = None

    # for patching interventions
    pos: Optional[int] = None
    code_pos: Optional[int] = -1

    # for description & regex-based interpretation
    description: Optional[str] = None
    regex: Optional[str] = None
    prec: Optional[float] = None
    recall: Optional[float] = None
    num_acts: Optional[int] = None

    def __post_init__(self):
        """Convert to appropriate types."""
        self.code = int(self.code)
        self.layer = int(self.layer)
        if self.head:
            self.head = int(self.head)
        if self.pos:
            self.pos = int(self.pos)
        if self.code_pos:
            self.code_pos = int(self.code_pos)
        if self.prec:
            self.prec = float(self.prec)
            assert 0 <= self.prec <= 1
        if self.recall:
            self.recall = float(self.recall)
            assert 0 <= self.recall <= 1
        if self.num_acts:
            self.num_acts = int(self.num_acts)

    def check_description_info(self):
        """Check if the regex info is present."""
        assert self.num_acts is not None and self.description is not None
        if self.regex is not None:
            assert self.prec is not None and self.recall is not None

    def check_patch_info(self):
        """Check if the patch info is present."""
        # TODO: pos can be none for patching
        assert self.pos is not None and self.code_pos is not None

    def __repr__(self):
        """Return the string representation."""
        repr = f"CodeInfo(code={self.code}, layer={self.layer}, head={self.head}, cb_at={self.cb_at}"
        if self.pos is not None or self.code_pos is not None:
            repr += f", pos={self.pos}, code_pos={self.code_pos}"
        if self.description is not None:
            repr += f", description={self.description}"
        if self.regex is not None:
            repr += f", regex={self.regex}, prec={self.prec}, recall={self.recall}"
        if self.num_acts is not None:
            repr += f", num_acts={self.num_acts}"
        repr += ")"
        return repr


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


def features_to_tokens(cb_key, cb_acts, num_codes, code=None):
    """Return the set of token ids each codebook feature activates on."""
    codebook_ids = cb_acts[cb_key]

    if code is None:
        features_tokens = [[] for _ in range(num_codes)]
        for i in tqdm(range(codebook_ids.shape[0])):
            for j in range(codebook_ids.shape[1]):
                for k in range(codebook_ids.shape[2]):
                    features_tokens[codebook_ids[i, j, k]].append((i, j))
    else:
        idx0, idx1, _ = np.where(codebook_ids == code)
        features_tokens = list(zip(idx0, idx1))

    return features_tokens


def color_str(s: str, html: bool, color: Optional[str] = None):
    """Color the string for html or terminal."""

    if html:
        color = "DeepSkyBlue" if color is None else color
        return f"<span style='color:{color}'>{s}</span>"
    else:
        color = "light_cyan" if color is None else color
        return colored(s, color)


def color_tokens_automata(tokens, color_idx, html=False):
    """Separate states with a dash and color red the tokens in color_idx."""
    ret_string = ""
    itr_over_color_idx = 0
    tokens_enumerate = enumerate(tokens)
    if tokens[0] == "<|endoftext|>":
        next(tokens_enumerate)
        if color_idx[0] == 0:
            itr_over_color_idx += 1
    for i, c in tokens_enumerate:
        if i % 2 == 1:
            ret_string += "-"
        if itr_over_color_idx < len(color_idx) and i == color_idx[itr_over_color_idx]:
            ret_string += color_str(c, html)
            itr_over_color_idx += 1
        else:
            ret_string += c
    return ret_string


def color_tokens(tokens, color_idx, n=3, html=False):
    """Color the tokens in color_idx."""
    ret_string = ""
    last_colored_token_idx = -1
    for i in color_idx:
        c_str = tokens[i]
        if i <= last_colored_token_idx + 2 * n + 1:
            ret_string += "".join(tokens[last_colored_token_idx + 1 : i])
        else:
            ret_string += "".join(
                tokens[last_colored_token_idx + 1 : last_colored_token_idx + n + 1]
            )
            ret_string += " ... "
            ret_string += "".join(tokens[i - n : i])
        ret_string += color_str(c_str, html)
        last_colored_token_idx = i
    ret_string += "".join(
        tokens[
            last_colored_token_idx + 1 : min(last_colored_token_idx + n, len(tokens))
        ]
    )
    return ret_string


def prepare_example_print(
    example_id,
    example_tokens,
    tokens_to_color,
    html,
    color_fn=color_tokens,
):
    """Format example to print."""
    example_output = color_str(example_id, html, "green")
    example_output += (
        ": "
        + color_fn(example_tokens, tokens_to_color, html=html)
        + ("<br>" if html else "\n")
    )
    return example_output


def tkn_print(
    ll,
    tokens,
    is_automata=False,
    n=3,
    max_examples=100,
    randomize=False,
    html=False,
    return_example_list=False,
):
    """Format and prints the tokens in ll."""
    if randomize:
        raise NotImplementedError("Randomize not yet implemented.")
    indices = range(len(ll))
    print_output = [] if return_example_list else ""
    curr_ex = ll[0][0]
    total_examples = 0
    tokens_to_color = []
    color_fn = color_tokens_automata if is_automata else partial(color_tokens, n=n)
    for idx in indices:
        if total_examples > max_examples:
            break
        i, j = ll[idx]

        if i != curr_ex and curr_ex >= 0:
            curr_ex_output = prepare_example_print(
                curr_ex,
                tokens[curr_ex],
                tokens_to_color,
                html,
                color_fn,
            )
            total_examples += 1
            if return_example_list:
                print_output.append((curr_ex_output, len(tokens_to_color)))
            else:
                print_output += curr_ex_output
            curr_ex = i
            tokens_to_color = []
        tokens_to_color.append(j)
    curr_ex_output = prepare_example_print(
        curr_ex,
        tokens[curr_ex],
        tokens_to_color,
        html,
        color_fn,
    )
    if return_example_list:
        print_output.append((curr_ex_output, len(tokens_to_color)))
    else:
        print_output += curr_ex_output
        asterisk_str = "********************************************"
        print_output += color_str(asterisk_str, html, "green")
    total_examples += 1

    return print_output


def print_ft_tkns(
    ft_tkns,
    tokens,
    is_automata=False,
    n=3,
    start=0,
    stop=1000,
    indices=None,
    max_examples=100,
    freq_filter=None,
    randomize=False,
    html=False,
    return_example_list=False,
):
    """Print the tokens for the codebook features."""
    indices = list(range(start, stop)) if indices is None else indices
    num_tokens = len(tokens) * len(tokens[0])
    codes, token_act_freqs, token_acts = [], [], []
    for i in indices:
        tkns = ft_tkns[i]
        freq = (len(tkns), 100 * len(tkns) / num_tokens)
        if freq_filter is not None and freq[1] > freq_filter:
            continue
        codes.append(i)
        token_act_freqs.append(freq)
        if len(tkns) > 0:
            tkn_acts = tkn_print(
                tkns,
                tokens,
                is_automata,
                n=n,
                max_examples=max_examples,
                randomize=randomize,
                html=html,
                return_example_list=return_example_list,
            )
            token_acts.append(tkn_acts)
        else:
            token_acts.append("")
    return codes, token_act_freqs, token_acts


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


def run_with_codes(
    input,
    cb_model,
    code,
    cb_at,
    layer_idx,
    head_idx=None,
    pos=None,
    code_pos=None,
    prepend_bos=True,
):
    """Run the model with the codebook features patched in."""
    hook_fns = [
        partial(patch_in_codes, pos=pos, code=code[i], code_pos=code_pos)
        for i in range(len(code))
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
        partial(patch_in_codes, pos=tupl.pos, code=tupl.code, code_pos=tupl.code_pos)
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
    """Get the common code in the cache."""
    codes, counts = torch.unique(cache_codes, return_counts=True, sorted=True)
    counts = counts.float() * 100
    counts /= cache_codes.shape[1]
    counts, indices = torch.sort(counts, descending=True)
    codes = codes[indices]
    indices = counts > threshold
    codes, counts = codes[indices], counts[indices]
    return codes, counts


def parse_code_info_string(
    info_str: str, cb_at="attn", pos=None, code_pos=-1
) -> CodeInfo:
    """Parse the code info string.

    The format of the `info_str` is:
    `code: 0, layer: 0, head: 0, occ_freq: 0.0, train_act_freq: 0.0`.
    """
    code, layer, head, occ_freq, train_act_freq = info_str.split(", ")
    code = int(code.split(": ")[1])
    layer = int(layer.split(": ")[1])
    head = int(head.split(": ")[1]) if head else None
    occ_freq = float(occ_freq.split(": ")[1])
    train_act_freq = float(train_act_freq.split(": ")[1])
    return CodeInfo(code, layer, head, pos=pos, code_pos=code_pos, cb_at=cb_at)


def parse_concept_codes_string(info_str: str, pos=None, code_append=False):
    """Parse the concept codes string."""
    code_info_strs = info_str.strip().split("\n")
    concept_codes = []
    layer, head = None, None
    code_pos = "append" if code_append else -1
    for code_info_str in code_info_strs:
        concept_codes.append(
            parse_code_info_string(code_info_str, pos=pos, code_pos=code_pos)
        )
        if code_append:
            continue
        if layer == concept_codes[-1].layer and head == concept_codes[-1].head:
            code_pos -= 1
        else:
            code_pos = -1
        concept_codes[-1].code_pos = code_pos
        layer, head = concept_codes[-1].layer, concept_codes[-1].head
    return concept_codes
