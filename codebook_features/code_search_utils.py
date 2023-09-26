"""Functions to help with searching codes using regex."""

import pickle
import re

import numpy as np
import torch
from tqdm import tqdm


def load_dataset_cache(cache_base_path):
    """Load cache files required for dataset from `cache_base_path`."""
    tokens_str = np.load(cache_base_path + "tokens_str.npy")
    tokens_text = np.load(cache_base_path + "tokens_text.npy")
    token_byte_pos = np.load(cache_base_path + "token_byte_pos.npy")
    return tokens_str, tokens_text, token_byte_pos


def load_code_search_cache(cache_base_path):
    """Load cache files required for code search from `cache_base_path`."""
    metrics = np.load(cache_base_path + "metrics.npy", allow_pickle=True).item()
    with open(cache_base_path + "cb_acts.pkl", "rb") as f:
        cb_acts = pickle.load(f)
    with open(cache_base_path + "act_count_ft_tkns.pkl", "rb") as f:
        act_count_ft_tkns = pickle.load(f)

    return cb_acts, act_count_ft_tkns, metrics


def search_re(re_pattern, tokens_text, at_odd_even=-1):
    """Get list of (example_id, token_pos) where re_pattern matches in tokens_text."""
    # TODO: ensure that parentheses are not escaped
    if re_pattern.find("(") == -1:
        re_pattern = f"({re_pattern})"
    res = [
        (i, finditer.span(1)[0])
        for i, text in enumerate(tokens_text)
        for finditer in re.finditer(re_pattern, text)
        if finditer.span(1)[0] != finditer.span(1)[1]
    ]
    if at_odd_even != -1:
        res = [r for r in res if r[1] % 2 == at_odd_even]
    return res


def byte_id_to_token_pos_id(example_byte_id, token_byte_pos):
    """Get (example_id, token_pos_id) for given (example_id, byte_id)."""
    example_id, byte_id = example_byte_id
    index = np.searchsorted(token_byte_pos[example_id], byte_id, side="right")
    return (example_id, index)


def get_code_pr(token_pos_ids, codebook_acts, cb_act_counts=None):
    """Get codes, prec, recall for given token_pos_ids and codebook_acts."""
    codes = np.array(
        [
            codebook_acts[example_id][token_pos_id]
            for example_id, token_pos_id in token_pos_ids
        ]
    )
    codes, counts = np.unique(codes, return_counts=True)
    recall = counts / len(token_pos_ids)
    idx = recall > 0.01
    codes, counts, recall = codes[idx], counts[idx], recall[idx]
    if cb_act_counts is not None:
        code_acts = np.array([cb_act_counts[code] for code in codes])
        prec = counts / code_acts
        sort_idx = np.argsort(prec)[::-1]
    else:
        code_acts = np.zeros_like(codes)
        prec = np.zeros_like(codes)
        sort_idx = np.argsort(recall)[::-1]
    codes, prec, recall = codes[sort_idx], prec[sort_idx], recall[sort_idx]
    code_acts = code_acts[sort_idx]
    return codes, prec, recall, code_acts


def get_neuron_pr(
    token_pos_ids, recall, neuron_acts_by_ex, neuron_sorted_acts, topk=10
):
    """Get codes, prec, recall for given token_pos_ids and codebook_acts."""
    if isinstance(neuron_acts_by_ex, torch.Tensor):
        re_neuron_acts = torch.stack(
            [
                neuron_acts_by_ex[example_id, token_pos_id]
                for example_id, token_pos_id in token_pos_ids
            ],
            dim=-1,
        )  # (layers, 2, dim_size, matches)
        re_neuron_acts = torch.sort(re_neuron_acts, dim=-1).values
    else:
        re_neuron_acts = np.stack(
            [
                neuron_acts_by_ex[example_id, token_pos_id]
                for example_id, token_pos_id in token_pos_ids
            ],
            axis=-1,
        )  # (layers, 2, dim_size, matches)
        re_neuron_acts.sort(axis=-1)
        re_neuron_acts = torch.from_numpy(re_neuron_acts)
    act_thresh = re_neuron_acts[:, :, :, -int(recall * re_neuron_acts.shape[-1])]
    assert neuron_sorted_acts.shape[:-1] == act_thresh.shape
    prec_den = torch.searchsorted(neuron_sorted_acts, act_thresh.unsqueeze(-1))
    prec_den = prec_den.squeeze(-1)
    prec_den = neuron_sorted_acts.shape[-1] - prec_den
    prec = int(recall * re_neuron_acts.shape[-1]) / prec_den
    assert (
        prec.shape == re_neuron_acts.shape[:-1]
    ), f"{prec.shape} != {re_neuron_acts.shape[:-1]}"

    best_neuron_idx = np.unravel_index(prec.argmax(), prec.shape)
    best_prec = prec[best_neuron_idx]
    best_neuron_act_thresh = act_thresh[best_neuron_idx].item()
    best_neuron_acts = neuron_acts_by_ex[
        :, :, best_neuron_idx[0], best_neuron_idx[1], best_neuron_idx[2]
    ]
    best_neuron_acts = best_neuron_acts >= best_neuron_act_thresh
    best_neuron_acts = np.stack(np.where(best_neuron_acts), axis=-1)

    return best_prec, best_neuron_acts, best_neuron_idx


def convert_to_adv_name(name, cb_at, ccb=""):
    """Convert layer0_head0 to layer0_attn_preproj_ccb0."""
    if ccb:
        layer, head = name.split("_")
        return layer + f"_{cb_at}_ccb" + head[4:]
    else:
        return layer + "_" + cb_at


def convert_to_base_name(name, ccb=""):
    """Convert layer0_attn_preproj_ccb0 to layer0_head0."""
    split_name = name.split("_")
    layer, head = split_name[0], split_name[-1][3:]
    if "ccb" in name:
        return layer + "_head" + head
    else:
        return layer


def get_layer_head_from_base_name(name):
    """Convert layer0_head0 to 0, 0."""
    split_name = name.split("_")
    layer = int(split_name[0][5:])
    head = None
    if len(split_name) > 1:
        head = int(split_name[-1][4:])
    return layer, head


def get_layer_head_from_adv_name(name):
    """Convert layer0_attn_preproj_ccb0 to 0, 0."""
    base_name = convert_to_base_name(name)
    layer, head = get_layer_head_from_base_name(base_name)
    return layer, head


def get_codes_from_pattern(
    re_pattern,
    tokens_text,
    token_byte_pos,
    cb_acts,
    act_count_ft_tkns,
    ccb="",
    topk=5,
    prec_threshold=0.5,
    at_odd_even=-1,
):
    """Fetch codes from a given regex pattern."""
    byte_ids = search_re(re_pattern, tokens_text, at_odd_even=at_odd_even)
    token_pos_ids = [
        byte_id_to_token_pos_id(ex_byte_id, token_byte_pos) for ex_byte_id in byte_ids
    ]
    token_pos_ids = np.unique(token_pos_ids, axis=0)
    re_token_matches = len(token_pos_ids)
    codebook_wise_codes = {}
    for cb_name, cb in tqdm(cb_acts.items()):
        base_cb_name = convert_to_base_name(cb_name, ccb=ccb)
        codes, prec, recall, code_acts = get_code_pr(
            token_pos_ids,
            cb,
            cb_act_counts=act_count_ft_tkns[base_cb_name],
        )
        idx = np.arange(min(topk, len(codes)))
        idx = idx[prec[:topk] > prec_threshold]
        codes, prec, recall = codes[idx], prec[idx], recall[idx]
        code_acts = code_acts[idx]
        codes_pr = list(zip(codes, prec, recall, code_acts))
        codebook_wise_codes[base_cb_name] = codes_pr
    return codebook_wise_codes, re_token_matches


def get_neurons_from_pattern(
    re_pattern,
    tokens_text,
    token_byte_pos,
    neuron_acts_by_ex,
    neuron_sorted_acts,
    recall_threshold,
    at_odd_even=-1,
):
    """Fetch the best neuron (with act thresh given by recall) from a given regex pattern."""
    byte_ids = search_re(re_pattern, tokens_text, at_odd_even=at_odd_even)
    token_pos_ids = [
        byte_id_to_token_pos_id(ex_byte_id, token_byte_pos) for ex_byte_id in byte_ids
    ]
    token_pos_ids = np.unique(token_pos_ids, axis=0)
    re_token_matches = len(token_pos_ids)
    best_prec, best_neuron_acts, best_neuron_idx = get_neuron_pr(
        token_pos_ids,
        recall_threshold,
        neuron_acts_by_ex,
        neuron_sorted_acts,
    )
    return best_prec, best_neuron_acts, best_neuron_idx, re_token_matches


def compare_codes_with_neurons(
    best_codes_info,
    tokens_text,
    token_byte_pos,
    neuron_acts_by_ex,
    neuron_sorted_acts,
    at_odd_even=-1,
):
    """Compare codes with neurons."""
    assert isinstance(neuron_acts_by_ex, np.ndarray)
    (
        all_best_prec,
        all_best_neuron_acts,
        all_best_neuron_idxs,
        all_re_token_matches,
    ) = zip(
        *[
            get_neurons_from_pattern(
                code_info.regex,
                tokens_text,
                token_byte_pos,
                neuron_acts_by_ex,
                neuron_sorted_acts,
                code_info.recall,
                at_odd_even=at_odd_even,
            )
            for code_info in tqdm(best_codes_info)
        ],
        strict=True,
    )
    all_best_prec = np.array(all_best_prec)
    code_best_precs = np.array([code_info.prec for code_info in best_codes_info])
    codes_better_than_neurons = code_best_precs > all_best_prec
    return codes_better_than_neurons.mean(), code_best_precs, np.array(all_best_prec)
