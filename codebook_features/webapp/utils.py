"""Utility functions for running webapp using streamlit."""

from typing import Optional

import numpy as np
import streamlit as st
from streamlit.components.v1 import html

from codebook_features import code_search_utils, utils

_PERSIST_STATE_KEY = f"{__name__}_PERSIST"
TOTAL_SAVE_BUTTONS = 0


def persist(key: str) -> str:
    """Mark widget state as persistent."""
    if _PERSIST_STATE_KEY not in st.session_state:
        st.session_state[_PERSIST_STATE_KEY] = set()

    st.session_state[_PERSIST_STATE_KEY].add(key)

    return key


def load_widget_state():
    """Load persistent widget state."""
    if _PERSIST_STATE_KEY in st.session_state:
        st.session_state.update(
            {
                key: value
                for key, value in st.session_state.items()
                if key in st.session_state[_PERSIST_STATE_KEY]
            }
        )


@st.cache_resource
def load_dataset_cache(dataset_cache_path):
    """Load cache files required for dataset from `cache_path`."""
    return code_search_utils.load_dataset_cache(dataset_cache_path)


@st.cache_resource
def load_code_search_cache(codes_cache_path, dataset_cache_path):
    """Load cache files required for code search from `codes_cache_path`."""
    (
        tokens_str,
        tokens_text,
        token_byte_pos,
    ) = load_dataset_cache(dataset_cache_path)
    (
        cb_acts,
        act_count_ft_tkns,
        metrics,
    ) = code_search_utils.load_code_search_cache(codes_cache_path)
    return tokens_str, tokens_text, token_byte_pos, cb_acts, act_count_ft_tkns, metrics


@st.cache_data(max_entries=100)
def load_ft_tkns(model_id, layer, head=None, code=None, topk=1):
    """Load the code-to-token map for a codebook."""
    # model_id required to not mix cache_data for different models
    assert model_id is not None
    cb_at = st.session_state["cb_at"]
    gcb = st.session_state["gcb"]
    cb_acts = st.session_state["cb_acts"]
    if head is not None:
        cb_name = f"layer{layer}_{cb_at}{gcb}{head}"
    else:
        cb_name = f"layer{layer}_{cb_at}"
    return utils.features_to_tokens(
        cb_name,
        cb_acts,
        num_codes=st.session_state["num_codes"],
        code=code,
        topk=topk,
    )


def get_code_acts(
    model_id,
    tokens_str,
    code,
    layer,
    head=None,
    ctx_size=5,
    num_examples=100,
    topk=1,
    return_example_list=False,
    is_fsm=False,
):
    """Get the token activations for a given code."""
    ft_tkns = load_ft_tkns(model_id, layer, head, code, topk=topk)
    ft_tkns = [ft_tkns]
    _, freqs, acts = utils.print_token_activations_of_codes(
        ft_tkns,
        tokens=tokens_str,
        indices=[0],
        html=True,
        n=ctx_size,
        max_examples=num_examples,
        return_example_list=return_example_list,
        is_fsm=is_fsm,
    )
    return acts[0], freqs[0]


def set_ct_acts(code, layer, head=None, extra_args=None, multiple_cbs=False):
    """Set the code and layer for the token activations."""
    # convert to int
    code, layer, head = int(code), int(layer), int(head) if head is not None else None
    st.session_state["ct_act_code"] = code
    st.session_state["ct_act_layer"] = layer
    if multiple_cbs:
        st.session_state["ct_act_head"] = head
    st.session_state["filter_codes"] = False

    info_txt = (
        f"layer: {layer},{f' head: {head},' if head is not None else ''} code: {code}"
    )
    if extra_args:
        for k, v in extra_args.items():
            info_txt += f", {k}: {v}"
    # to copy info_txt to clipboard of the user
    my_html = f"""
    <script>
        async function myF() {{
            await new Promise(r => setTimeout(r, 10));
            const textarea = document.createElement("textarea");
            textarea.textContent = "{info_txt}";
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand("copy");
            document.body.removeChild(textarea);
        }}
        myF();
        document.location.href = "#code-token-activations";

    </script>
    """
    html(my_html, height=0, width=0, scrolling=False)


def find_next_code(code, layer_code_acts, act_range=None):
    """Find the next code that has activations in the given range."""
    if act_range is None:
        return code
    min_act, max_act = 0, np.inf
    if isinstance(act_range, tuple):
        if len(act_range) == 2:
            min_act, max_act = act_range
        else:
            min_act = act_range[0]
    elif isinstance(act_range, int):
        min_act = act_range
    for code_iter, code_act_count in enumerate(layer_code_acts[code:]):
        if code_act_count >= min_act and code_act_count <= max_act:
            code += code_iter
            break
    return code


def escape_markdown(text):
    """Escapes markdown special characters."""
    MD_SPECIAL_CHARS = r"\`*_{}[]()#+-.!$"
    for char in MD_SPECIAL_CHARS:
        text = text.replace(char, "\\" + char)
    return text


def add_code_to_demo_file(code_info: utils.CodeInfo, file_path: str):
    """Add code to demo file."""
    # TODO: add check for duplicate code and return False if found
    # TODO: convert saved codes to databases instead of txt files?
    code_info.check_description_info()
    with open(file_path, "a") as f:
        f.write("\n")
        f.write(f"# {code_info.description}:")
        if code_info.regex:
            f.write(f" {code_info.regex}")
        f.write("\n")
        f.write(f"layer: {code_info.layer}")
        f.write(f", head: {code_info.head}" if code_info.head is not None else "")
        f.write(f", code: {code_info.code}")
        if code_info.regex:
            f.write(f", prec: {code_info.prec:.4f}, recall: {code_info.recall:.4f}")
        f.write(f", num_acts: {code_info.num_acts}\n")
        return True


def add_save_code_button(
    demo_file_path: str,
    num_acts: int,
    save_regex: bool = False,
    prec: Optional[float] = None,
    recall: Optional[float] = None,
    button_st_container=st,
    button_text: bool = False,
    button_key_suffix: str = "",
):
    """Add a button on streamlit to save code to demo codes file."""
    save_button = button_st_container.button(
        "💾" + (" Save Code to Demos" if button_text else ""),
        key=f"save_code_button{button_key_suffix}",
        help="Save code to demo codes file",
    )
    if save_button:
        description = st.text_input(
            "Write a description for the code",
            key=f"save_code_desc{button_key_suffix}",
        )
        if not description:
            return

    description = st.session_state.get(f"save_code_desc{button_key_suffix}", None)
    if description:
        layer = st.session_state["ct_act_layer"]
        multiple_cbs = st.session_state["multiple_cbs"]
        if multiple_cbs:
            head = st.session_state["ct_act_head"]
        else:
            head = None

        code = st.session_state["ct_act_code"]
        code_info = utils.CodeInfo(
            layer=layer,
            head=head,
            code=code,
            description=description,
            num_acts=num_acts,
        )

        if save_regex:
            code_info.regex = st.session_state["regex_pattern"]
            code_info.prec = prec
            code_info.recall = recall

        saved = add_code_to_demo_file(code_info, demo_file_path)
        if saved:
            st.success("Code saved!", icon="🎉")
