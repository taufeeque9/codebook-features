"""Util functions for codebook features."""

import pathlib
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

    @classmethod
    def from_str(cls, code_txt, *args, **kwargs):
        """Extract code info fields from string."""
        code_txt = code_txt.strip().lower()
        code_txt = code_txt.split(", ")
        code_txt = dict(txt.split(": ") for txt in code_txt)
        return cls(*args, **code_txt, **kwargs)


@dataclass
class ModelInfoForWebapp:
    """Model info for webapp."""

    model_name: str
    pretrained_path: str
    dataset_name: str
    num_codes: int
    cb_at: str
    gcb: str
    n_layers: int
    n_grps: Optional[int] = None
    seed: int = 42
    max_samples: int = 2000

    def __post_init__(self):
        """Convert to correct types."""
        self.num_codes = int(self.num_codes)
        self.n_layers = int(self.n_layers)
        if self.n_grps == "None":
            self.n_grps = None
        elif self.n_grps is not None:
            self.n_grps = int(self.n_grps)
        self.seed = int(self.seed)
        self.max_samples = int(self.max_samples)

    @classmethod
    def load(cls, path):
        """Parse model info from path."""
        path = pathlib.Path(path)
        with open(path / "info.txt", "r") as f:
            lines = f.readlines()
            lines = dict(line.strip().split(": ") for line in lines)
        return cls(**lines)

    def save(self, path):
        """Save model info to path."""
        path = pathlib.Path(path)
        with open(path / "info.txt", "w") as f:
            for k, v in self.__dict__.items():
                f.write(f"{k}: {v}\n")


def logits_to_pred(logits, tokenizer, k=5):
    """Convert logits to top-k predictions."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = sorted_logits.softmax(dim=-1)
    topk_preds = [tokenizer.convert_ids_to_tokens(e) for e in sorted_indices[:, -1, :k]]
    topk_preds = [tokenizer.convert_tokens_to_string([e]) for batch in topk_preds for e in batch]
    return [(topk_preds[i], probs[:, -1, i].item()) for i in range(len(topk_preds))]


def features_to_tokens(cb_key, cb_acts, num_codes, code=None, topk=1):
    """Return the set of token ids each codebook feature activates on."""
    codebook_ids = cb_acts[cb_key]

    if code is None:
        features_tokens = [[] for _ in range(num_codes)]
        for i in tqdm(range(codebook_ids.shape[0])):
            for j in range(codebook_ids.shape[1]):
                for k in range(codebook_ids.shape[2]):
                    features_tokens[codebook_ids[i, j, k]].append((i, j))
    else:
        idx0, idx1, _ = np.where(codebook_ids[:, :, :topk] == code)
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


def color_tokens_tokfsm(tokens, color_idx, html=False):
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
            ret_string += "".join(tokens[last_colored_token_idx + 1 : last_colored_token_idx + n + 1])
            ret_string += " ... "
            ret_string += "".join(tokens[i - n : i])
        ret_string += color_str(c_str, html)
        last_colored_token_idx = i
    ret_string += "".join(tokens[last_colored_token_idx + 1 : min(last_colored_token_idx + n, len(tokens))])
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
    example_output += ": " + color_fn(example_tokens, tokens_to_color, html=html) + ("<br>" if html else "\n")
    return example_output


def print_token_activations_of_code(
    code_act_by_pos,
    tokens,
    is_fsm=False,
    n=3,
    max_examples=100,
    randomize=False,
    html=False,
    return_example_list=False,
):
    """Print the context with the tokens that a code activates on.

    Args:
        code_act_by_pos: list of (example_id, token_pos_id) tuples specifying
            the token positions that a code activates on in a dataset.
        tokens: list of tokens of a dataset.
        is_fsm: whether the dataset is the TokFSM dataset.
        n: context to print around each side of a token that the code activates on.
        max_examples: maximum number of examples to print.
        randomize: whether to randomize the order of examples.
        html: Format the printing style for html or terminal.
        return_example_list: whether to return the printed string by examples or as a single string.

    Returns:
        string of all examples formatted if `return_example_list` is False otherwise
        list of (example_string, num_tokens_colored) tuples for each example.
    """
    if randomize:
        raise NotImplementedError("Randomize not yet implemented.")
    indices = range(len(code_act_by_pos))
    print_output = [] if return_example_list else ""
    curr_ex = code_act_by_pos[0][0]
    total_examples = 0
    tokens_to_color = []
    color_fn = color_tokens_tokfsm if is_fsm else partial(color_tokens, n=n)
    for idx in indices:
        if total_examples > max_examples:
            break
        i, j = code_act_by_pos[idx]

        if i != curr_ex and curr_ex >= 0:
            # got new example so print the previous one
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
        print_output += color_str("*" * 50, html, "green")
    total_examples += 1

    return print_output


def print_token_activations_of_codes(
    ft_tkns,
    tokens,
    is_fsm=False,
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
        tkns_of_code = ft_tkns[i]
        freq = (len(tkns_of_code), 100 * len(tkns_of_code) / num_tokens)
        if freq_filter is not None and freq[1] > freq_filter:
            continue
        codes.append(i)
        token_act_freqs.append(freq)
        if len(tkns_of_code) > 0:
            tkn_acts = print_token_activations_of_code(
                tkns_of_code,
                tokens,
                is_fsm,
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


def get_cb_hook_key(cb_at: str, layer_idx: int, gcb_idx: Optional[int] = None):
    """Get the layer name used to store hooks/cache."""
    comp_name = "attn" if "attn" in cb_at else "mlp"
    if gcb_idx is None:
        return f"blocks.{layer_idx}.{comp_name}.codebook_layer.hook_codebook_ids"
    else:
        return f"blocks.{layer_idx}.{comp_name}.codebook_layer.codebook.{gcb_idx}.hook_codebook_ids"


def run_model_fn_with_codes(
    input,
    cb_model,
    fn_name,
    fn_kwargs=None,
    list_of_code_infos=(),
):
    """Run the `cb_model`'s `fn_name` method while activating the codes in `list_of_code_infos`.

    Common use case includes running the `run_with_cache` method while activating the codes.
    For running the `generate` method, use `generate_with_codes` instead.
    """
    if fn_kwargs is None:
        fn_kwargs = {}
    hook_fns = [
        partial(patch_in_codes, pos=tupl.pos, code=tupl.code, code_pos=tupl.code_pos) for tupl in list_of_code_infos
    ]
    fwd_hooks = [
        (get_cb_hook_key(tupl.cb_at, tupl.layer, tupl.head), hook_fns[i]) for i, tupl in enumerate(list_of_code_infos)
    ]
    cb_model.reset_hook_kwargs()
    with cb_model.hooks(fwd_hooks, [], True, False) as hooked_model:
        ret = hooked_model.__getattribute__(fn_name)(input, **fn_kwargs)
    return ret


def generate_with_codes(
    input,
    cb_model,
    list_of_code_infos=(),
    tokfsm=None,
    generate_kwargs=None,
):
    """Sample from the language model while activating the codes in `list_of_code_infos`."""
    gen = run_model_fn_with_codes(
        input,
        cb_model,
        "generate",
        generate_kwargs,
        list_of_code_infos,
    )
    return tokfsm.seq_to_traj(gen) if tokfsm is not None else gen


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


def cb_hook_key_to_info(layer_hook_key: str):
    """Get the layer info from the codebook layer hook key.

    Args:
        layer_hook_key: the hook key of the codebook layer.
            E.g. `blocks.3.attn.codebook_layer.hook_codebook_ids`

    Returns:
        comp_name: the name of the component codebook is applied at.
        layer_idx: the layer index.
        gcb_idx: the codebook index if the codebook layer is grouped, otherwise None.
    """
    layer_search = re.search(r"blocks\.(\d+)\.(\w+)\.", layer_hook_key)
    assert layer_search is not None
    layer_idx, comp_name = int(layer_search.group(1)), layer_search.group(2)
    gcb_idx_search = re.search(r"codebook\.(\d+)", layer_hook_key)
    if gcb_idx_search is not None:
        gcb_idx = int(gcb_idx_search.group(1))
    else:
        gcb_idx = None
    return comp_name, layer_idx, gcb_idx


def find_code_changes(cache1, cache2, pos=None):
    """Find the codebook codes that are different between the two caches."""
    for k in cache1.keys():
        if "codebook" in k:
            c1 = cache1[k][0, pos]
            c2 = cache2[k][0, pos]
            if not torch.all(c1 == c2):
                print(cb_hook_key_to_info(k), c1.tolist(), c2.tolist())
                print(cb_hook_key_to_info(k), c1.tolist(), c2.tolist())


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


def parse_topic_codes_string(
    info_str: str,
    pos: Optional[int] = None,
    code_append: Optional[bool] = False,
    **code_info_kwargs,
):
    """Parse the topic codes string."""
    code_info_strs = info_str.strip().split("\n")
    code_info_strs = [e.strip() for e in code_info_strs if e]
    topic_codes = []
    layer, head = None, None
    if code_append is None:
        code_pos = None
    else:
        code_pos = "append" if code_append else -1
    for code_info_str in code_info_strs:
        topic_codes.append(
            CodeInfo.from_str(
                code_info_str,
                pos=pos,
                code_pos=code_pos,
                **code_info_kwargs,
            )
        )
        if code_append is None or code_append:
            continue
        if layer == topic_codes[-1].layer and head == topic_codes[-1].head:
            code_pos -= 1  # type: ignore
        else:
            code_pos = -1
        topic_codes[-1].code_pos = code_pos
        layer, head = topic_codes[-1].layer, topic_codes[-1].head
    return topic_codes


def find_similar_codes(cb_model, code_info, n=8):
    """Find the `n` most similar codes to the given code using cosine similarity.

    Useful for finding related codes for interpretability.
    """
    codebook = cb_model.get_codebook(code_info)
    device = codebook.weight.device
    code = codebook(torch.tensor(code_info.code).to(device))
    code = code.to(device)
    logits = torch.matmul(code, codebook.weight.T)
    _, indices = torch.topk(logits, n)
    assert indices[0] == code_info.code
    assert torch.allclose(logits[indices[0]], torch.tensor(1.0))
    return indices[1:], logits[indices[1:]].tolist()
