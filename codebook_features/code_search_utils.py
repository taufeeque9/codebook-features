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
    """Get list of (example_id, token_pos) where re_pattern matches in tokens_text.

    Args:
        re_pattern: regex pattern to search for.
        tokens_text: list of example texts.
        at_odd_even: to limit matches to odd or even positions only.
            -1 (default): to not limit matches.
            0: to limit matches to odd positions only.
            1: to limit matches to even positions only.
            This is useful for the TokFSM dataset when searching for states
            since the first token of states are always at even positions.
    """
    # TODO: ensure that parentheses are not escaped
    assert at_odd_even in [-1, 0, 1], f"Invalid at_odd_even: {at_odd_even}"
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
    """Convert byte position (or character position in a text) to its token position.

    Used to convert the searched regex span to its token position.

    Args:
        example_byte_id: tuple of (example_id, byte_id) where byte_id is a
            character's position in the text.
        token_byte_pos: numpy array of shape (num_examples, seq_len) where
            `token_byte_pos[example_id][token_pos]` is the byte position of
            the token at `token_pos` in the example with `example_id`.

    Returns:
        (example_id, token_pos_id) tuple.
    """
    example_id, byte_id = example_byte_id
    index = np.searchsorted(token_byte_pos[example_id], byte_id, side="right")
    return (example_id, index)


def get_code_precision_and_recall(token_pos_ids, codebook_acts, cb_act_counts=None):
    """Search for the codes that activate on the given `token_pos_ids`.

    Args:
        token_pos_ids: list of (example_id, token_pos_id) tuples.
        codebook_acts: numpy array of activations of a codebook on a dataset with
            shape (num_examples, seq_len, k_codebook).
        cb_act_counts: array of shape (num_codes,) where `cb_act_counts[cb_name][code]`
            is the number of times the code `code` is activated in the dataset.

    Returns:
        codes: numpy array of code ids sorted by their precision on the given `token_pos_ids`.
        prec: numpy array where `prec[i]` is the precision of the code
            `codes[i]` for the given `token_pos_ids`.
        recall: numpy array where `recall[i]` is the recall of the code
            `codes[i]` for the given `token_pos_ids`.
        code_acts: numpy array where `code_acts[i]` is the number of times
            the code `codes[i]` is activated in the dataset.
    """
    codes = np.array([codebook_acts[example_id, token_pos_id] for example_id, token_pos_id in token_pos_ids])
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


def get_neuron_precision_and_recall(token_pos_ids, recall, neuron_acts_by_ex, neuron_sorted_acts):
    """Get the neurons with the highest precision and recall for the given `token_pos_ids`.

    Args:
        token_pos_ids: list of token (example_id, token_pos_id) tuples from a dataset over which
            the neurons with the highest precision and recall are to be found.
        recall: recall threshold for the neurons (this determines their activation threshold).
        neuron_acts_by_ex: numpy array of activations of all the attention and mlp output neurons
            on a dataset with shape (num_examples, seq_len, num_layers, 2, dim_size).
            The third dimension is 2 because we consider neurons from both: attention and mlp.
        neuron_sorted_acts: numpy array of sorted activations of all the attention and mlp output neurons
            on a dataset with shape (num_layers, 2, dim_size, num_examples * seq_len).
            This should be obtained using the `neuron_acts_by_ex` array by rearranging the first two
            dimensions to the last dimensions and then sorting the last dimension.

    Returns:
        best_prec: highest precision amongst all the neurons for the given `token_pos_ids`.
        best_neuron_acts: number of activations of the best neuron for the given `token_pos_ids`
            based on the threshold determined by the `recall` argument.
        best_neuron_idx: tuple of (layer, is_mlp, neuron_id) where `layer` is the layer number,
            `is_mlp` is 0 if the neuron is from attention and 1 if the neuron is from mlp,
            and `neuron_id` is the neuron's index in the layer.
    """
    if isinstance(neuron_acts_by_ex, torch.Tensor):
        neuron_acts_on_pattern = torch.stack(
            [neuron_acts_by_ex[example_id, token_pos_id] for example_id, token_pos_id in token_pos_ids],
            dim=-1,
        )  # (layers, 2, dim_size, matches)
        neuron_acts_on_pattern = torch.sort(neuron_acts_on_pattern, dim=-1).values
    else:
        neuron_acts_on_pattern = np.stack(
            [neuron_acts_by_ex[example_id, token_pos_id] for example_id, token_pos_id in token_pos_ids],
            axis=-1,
        )  # (layers, 2, dim_size, matches)
        neuron_acts_on_pattern.sort(axis=-1)
        neuron_acts_on_pattern = torch.from_numpy(neuron_acts_on_pattern)
    act_thresh = neuron_acts_on_pattern[:, :, :, -int(recall * neuron_acts_on_pattern.shape[-1])]
    assert neuron_sorted_acts.shape[:-1] == act_thresh.shape
    prec_den = torch.searchsorted(neuron_sorted_acts, act_thresh.unsqueeze(-1))
    prec_den = prec_den.squeeze(-1)
    prec_den = neuron_sorted_acts.shape[-1] - prec_den
    prec = int(recall * neuron_acts_on_pattern.shape[-1]) / prec_den
    assert prec.shape == neuron_acts_on_pattern.shape[:-1], f"{prec.shape} != {neuron_acts_on_pattern.shape[:-1]}"

    best_neuron_idx = np.unravel_index(prec.argmax(), prec.shape)
    best_prec = prec[best_neuron_idx]
    best_neuron_act_thresh = act_thresh[best_neuron_idx].item()
    best_neuron_acts = neuron_acts_by_ex[:, :, best_neuron_idx[0], best_neuron_idx[1], best_neuron_idx[2]]
    best_neuron_acts = best_neuron_acts >= best_neuron_act_thresh
    best_neuron_acts = np.stack(np.where(best_neuron_acts), axis=-1)

    return best_prec, best_neuron_acts, best_neuron_idx


def convert_to_adv_name(name, cb_at, gcb=""):
    """Convert layer0_head0 to layer0_attn_preproj_gcb0."""
    if gcb:
        layer, head = name.split("_")
        return layer + f"_{cb_at}_gcb" + head[4:]
    else:
        return layer + "_" + cb_at


def convert_to_base_name(name, gcb=""):
    """Convert layer0_attn_preproj_gcb0 to layer0_head0."""
    split_name = name.split("_")
    layer, head = split_name[0], split_name[-1][3:]
    if "gcb" in name:
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
    """Convert layer0_attn_preproj_gcb0 to 0, 0."""
    base_name = convert_to_base_name(name)
    layer, head = get_layer_head_from_base_name(base_name)
    return layer, head


def get_codes_from_pattern(
    re_pattern,
    tokens_text,
    token_byte_pos,
    cb_acts,
    act_count_ft_tkns,
    gcb="",
    topk=5,
    prec_threshold=0.5,
    at_odd_even=-1,
):
    """Fetch codes that activate on a given regex pattern.

    Retrieves at most `top_k` codes that activate with precision above `prec_threshold`.

    Args:
        re_pattern: regex pattern to search for.
        tokens_text: list of example texts of a dataset.
        token_byte_pos: numpy array of shape (num_examples, seq_len) where
            `token_byte_pos[example_id][token_pos]` is the byte position of
            the token at `token_pos` in the example with `example_id`.
        cb_acts: dict of codebook activations.
        act_count_ft_tkns: dict over all codebooks of number of token activations on the dataset
        gcb: "_gcb" for grouped codebooks and "" for non-grouped codebooks.
        topk: maximum number of codes to return per codebook.
        prec_threshold: minimum precision required for a code to be returned.
        at_odd_even: to limit matches to odd or even positions only.
            -1 (default): to not limit matches.
            0: to limit matches to odd positions only.
            1: to limit matches to even positions only.
            This is useful for the TokFSM dataset when searching for states
            since the first token of states are always at even positions.

    Returns:
        codebook_wise_codes: dict of codebook name to list of
        (code, prec, recall, code_acts) tuples.
        re_token_matches: number of tokens that match the regex pattern.
    """
    byte_ids = search_re(re_pattern, tokens_text, at_odd_even=at_odd_even)
    token_pos_ids = [byte_id_to_token_pos_id(ex_byte_id, token_byte_pos) for ex_byte_id in byte_ids]
    token_pos_ids = np.unique(token_pos_ids, axis=0)
    re_token_matches = len(token_pos_ids)
    codebook_wise_codes = {}
    for cb_name, cb in tqdm(cb_acts.items()):
        base_cb_name = convert_to_base_name(cb_name, gcb=gcb)
        codes, prec, recall, code_acts = get_code_precision_and_recall(
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
    """Fetch the highest precision neurons that activate on a given regex pattern.

    The activation threshold for the neurons is determined by the `recall_threshold`.

    Args:
        re_pattern: regex pattern to search for.
        tokens_text: list of example texts of a dataset.
        token_byte_pos: numpy array of shape (num_examples, seq_len) where
            `token_byte_pos[example_id][token_pos]` is the byte position of
            the token at `token_pos` in the example with `example_id`.
        neuron_acts_by_ex: numpy array of activations of all the attention and mlp output neurons
            on a dataset with shape (num_examples, seq_len, num_layers, 2, dim_size).
            The third dimension is 2 because we consider neurons from both: attention and mlp.
        neuron_sorted_acts: numpy array of sorted activations of all the attention and mlp output neurons
            on a dataset with shape (num_layers, 2, dim_size, num_examples * seq_len).
            This should be obtained using the `neuron_acts_by_ex` array by rearranging the first two
            dimensions to the last dimensions and then sorting the last dimension.
        recall_threshold: recall threshold for the neurons (this determines their activation threshold).
        at_odd_even: to limit matches to odd or even positions only.
            -1 (default): to not limit matches.
            0: to limit matches to odd positions only.
            1: to limit matches to even positions only.
            This is useful for the TokFSM dataset when searching for states
            since the first token of states are always at even positions.

    Returns:
        best_prec: highest precision amongst all the neurons for the given `token_pos_ids`.
        best_neuron_acts: number of activations of the best neuron for the given `token_pos_ids`
            based on the threshold determined by the `recall` argument.
        best_neuron_idx: tuple of (layer, is_mlp, neuron_id) where `layer` is the layer number,
            `is_mlp` is 0 if the neuron is from attention and 1 if the neuron is from mlp,
            and `neuron_id` is the neuron's index in the layer.
        re_token_matches: number of tokens that match the regex pattern.
    """
    byte_ids = search_re(re_pattern, tokens_text, at_odd_even=at_odd_even)
    token_pos_ids = [byte_id_to_token_pos_id(ex_byte_id, token_byte_pos) for ex_byte_id in byte_ids]
    token_pos_ids = np.unique(token_pos_ids, axis=0)
    re_token_matches = len(token_pos_ids)
    best_prec, best_neuron_acts, best_neuron_idx = get_neuron_precision_and_recall(
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
    """Compare codes with the highest precision neurons on the regex pattern of the code.

    Args:
        best_codes_info: list of CodeInfo objects.
        tokens_text: list of example texts of a dataset.
        token_byte_pos: numpy array of shape (num_examples, seq_len) where
            `token_byte_pos[example_id][token_pos]` is the byte position of
            the token at `token_pos` in the example with `example_id`.
        neuron_acts_by_ex: numpy array of activations of all the attention and mlp output neurons
            on a dataset with shape (num_examples, seq_len, num_layers, 2, dim_size).
            The third dimension is 2 because we consider neurons from both: attention and mlp.
        neuron_sorted_acts: numpy array of sorted activations of all the attention and mlp output neurons
            on a dataset with shape (num_layers, 2, dim_size, num_examples * seq_len).
            This should be obtained using the `neuron_acts_by_ex` array by rearranging the first two
            dimensions to the last dimensions and then sorting the last dimension.
        at_odd_even: to limit matches to odd or even positions only.
            -1 (default): to not limit matches.
            0: to limit matches to odd positions only.
            1: to limit matches to even positions only.
            This is useful for the TokFSM dataset when searching for states
            since the first token of states are always at even positions.

    Returns:
        codes_better_than_neurons: fraction of codes that have higher precision than the highest
            precision neuron on the regex pattern of the code.
        code_best_precs: is an array of the precision of each code in `best_codes_info`.
        all_best_prec: is an array of the highest precision neurons on the regex pattern.
    """
    assert isinstance(neuron_acts_by_ex, np.ndarray)
    (
        neuron_best_prec,
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
    neuron_best_prec = np.array(neuron_best_prec)
    code_best_precs = np.array([code_info.prec for code_info in best_codes_info])
    codes_better_than_neurons = code_best_precs > neuron_best_prec
    return codes_better_than_neurons.mean(), code_best_precs, neuron_best_prec
