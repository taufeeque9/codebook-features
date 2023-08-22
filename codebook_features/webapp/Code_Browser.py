"""Web App for the Codebook Features project."""

import glob
import os

import streamlit as st

from codebook_features import code_search_utils
from codebook_features.webapp import utils as webapp_utils

DEPLOY_MODE = False


webapp_utils.load_widget_state()

st.set_page_config(
    page_title="Codebook Features",
    page_icon="📚",
)

st.title("Codebook Features")

base_cache_dir = "/shared/cb_eval_acts/"
dirs = glob.glob(base_cache_dir + "*/")
model_name_options = [d.split("/")[-2].split("_")[:-2] for d in dirs]
model_name_options = ["_".join(m) for m in model_name_options]
model_name_options = sorted(set(model_name_options))

model_name = st.selectbox(
    "Model",
    model_name_options,
    key=webapp_utils.persist("model_name"),
)

model = model_name.split("_")[0].split("#")[0]
model_layers = {
    "pythia-410m-deduped": 24,
    "pythia-70m-deduped": 6,
    "gpt2": 12,
    "TinyStories-1Layer-21M": 1,
}
model_heads = {
    "pythia-410m-deduped": 16,
    "pythia-70m-deduped": 8,
    "gpt2": 12,
    "TinyStories-1Layer-21M": 16,
}
ccb = model_name.split("_")[1]
ccb = "_ccb" if ccb == "ccb" else ""
cb_at = "_".join(model_name.split("_")[2:])
seq_len = 512 if "tinystories" in model_name.lower() else 1024
st.session_state["seq_len"] = seq_len

cache_path = "/shared/cb_eval_acts/" + model_name + "_*"
dirs = glob.glob(cache_path)
dirs.sort(key=os.path.getmtime)

# session states
is_attn = "attn" in cb_at
num_layers = model_layers[model]
num_heads = model_heads[model]
cache_path = dirs[-1] + "/"

model_info = code_search_utils.parse_model_info(cache_path)
num_codes = model_info.num_codes

(
    tokens_str,
    tokens_text,
    token_byte_pos,
    cb_acts,
    act_count_ft_tkns,
    metrics,
) = webapp_utils.load_code_search_cache(cache_path)
metric_keys = ["eval_loss", "eval_accuracy", "eval_dead_code_fraction"]
metrics = {k: v for k, v in metrics.items() if k.split("/")[0] in metric_keys}

st.session_state["model_name_id"] = model_name
st.session_state["cb_acts"] = cb_acts
st.session_state["tokens_text"] = tokens_text
st.session_state["tokens_str"] = tokens_str
st.session_state["act_count_ft_tkns"] = act_count_ft_tkns

st.session_state["num_codes"] = num_codes
st.session_state["ccb"] = ccb
st.session_state["cb_at"] = cb_at
st.session_state["is_attn"] = is_attn

st.markdown("## Metrics")
# hide metrics by default
if st.checkbox("Show Model Metrics"):
    st.write(metrics)

st.markdown("## Demo Codes")
demo_file_path = cache_path + "demo_codes.txt"

if st.checkbox("Show Demo Codes"):
    try:
        with open(demo_file_path, "r") as f:
            demo_codes = f.readlines()
    except FileNotFoundError:
        demo_codes = []

    code_desc, code_regex = "", ""
    demo_codes = [code.strip() for code in demo_codes if code.strip()]

    num_cols = 6 if is_attn else 5
    cols = st.columns([1] * (num_cols - 1) + [2])
    # st.markdown(button_height_style, unsafe_allow_html=True)
    cols[0].markdown("Search", help="Button to see token activations for the code.")
    cols[1].write("Code")
    cols[2].write("Layer")
    if is_attn:
        cols[3].write("Head")
    cols[-2].markdown(
        "Num Acts",
        help="Number of tokens that the code activates on in the acts dataset.",
    )
    cols[-1].markdown("Description", help="Interpreted description of the code.")

    if len(demo_codes) == 0:
        st.markdown(
            f"""
            <div style="font-size: 1.3rem; color: red;">
            No demo codes found in file {demo_file_path}
            </div>
            """,
            unsafe_allow_html=True,
        )
    skip = True
    for code_txt in demo_codes:
        if code_txt.startswith("##"):
            skip = True
            continue
        if code_txt.startswith("#"):
            code_desc, code_regex = code_txt[1:].split(":")
            code_desc, code_regex = code_desc.strip(), code_regex.strip()
            skip = False
            continue
        if skip:
            continue
        code_info = code_search_utils.get_code_info_pr_from_str(code_txt, code_regex)
        comp_info = f"layer{code_info.layer}_{f'head{code_info.head}' if code_info.head is not None else ''}"
        button_key = (
            f"demo_search_code{code_info.code}_layer{code_info.layer}_desc-{code_info.description}"
            + (f"head{code_info.head}" if code_info.head is not None else "")
        )
        cols = st.columns([1] * (num_cols - 1) + [2])
        button_clicked = cols[0].button(
            "🔍",
            key=button_key,
        )
        if button_clicked:
            webapp_utils.set_ct_acts(
                code_info.code, code_info.layer, code_info.head, None, is_attn
            )
        cols[1].write(code_info.code)
        cols[2].write(str(code_info.layer))
        if is_attn:
            cols[3].write(str(code_info.head))
        cols[-2].write(str(act_count_ft_tkns[comp_info][code_info.code]))
        cols[-1].write(code_desc)
        skip = True


st.markdown("## Code Search")

regex_pattern = st.text_input(
    "Enter a regex pattern",
    help="Wrap code token in the first group. E.g. New (York)",
    key="regex_pattern",
)
# topk = st.slider("Top K", 1, 20, 10)
prec_col, sort_col = st.columns(2)
prec_threshold = prec_col.slider(
    "Precision Threshold",
    0.0,
    1.0,
    0.9,
    help="Shows codes with precision on the regex pattern above the threshold.",
)
sort_by_options = ["Precision", "Recall", "Num Acts"]
sort_by_name = sort_col.radio(
    "Sort By",
    sort_by_options,
    index=0,
    horizontal=True,
    help="Sorts the codes by the selected metric.",
)
sort_by = sort_by_options.index(sort_by_name)


@st.cache_data(ttl=3600)
def get_codebook_wise_codes_for_regex(regex_pattern, prec_threshold, ccb, model_name):
    """Get codebook wise codes for a given regex pattern."""
    return code_search_utils.get_codes_from_pattern(
        regex_pattern,
        tokens_text,
        token_byte_pos,
        cb_acts,
        act_count_ft_tkns,
        ccb=ccb,
        topk=8,
        prec_threshold=prec_threshold,
    )


if regex_pattern:
    print(regex_pattern)
    print("raw:", repr(regex_pattern))
    codebook_wise_codes, re_token_matches = get_codebook_wise_codes_for_regex(
        regex_pattern,
        prec_threshold,
        ccb,
        model_name,
    )
    st.markdown(f"Found :green[{re_token_matches}] matches")
    num_search_cols = 7 if is_attn else 6
    non_deploy_offset = 0
    if not DEPLOY_MODE:
        non_deploy_offset = 1
        num_search_cols += non_deploy_offset

    cols = st.columns(num_search_cols)

    # st.markdown(button_height_style, unsafe_allow_html=True)

    cols[0].markdown("Search", help="Button to see token activations for the code.")
    cols[1].write("Layer")
    if is_attn:
        cols[2].write("Head")
    cols[-4 - non_deploy_offset].write("Code")
    cols[-3 - non_deploy_offset].write("Precision")
    cols[-2 - non_deploy_offset].write("Recall")
    cols[-1 - non_deploy_offset].markdown(
        "Num Acts",
        help="Number of tokens that the code activates on in the acts dataset.",
    )
    if not DEPLOY_MODE:
        cols[-1].markdown(
            "Save to Demos",
            help="Button to save the code to demos along with the regex pattern.",
        )
    all_codes = codebook_wise_codes.items()
    all_codes = [
        (cb_name, code_pr_info)
        for cb_name, code_pr_infos in all_codes
        for code_pr_info in code_pr_infos
    ]
    all_codes = sorted(all_codes, key=lambda x: x[1][1 + sort_by], reverse=True)
    for cb_name, (code, prec, rec, code_acts) in all_codes:
        layer_head = cb_name.split("_")
        layer = layer_head[0][5:]
        head = layer_head[1][4:] if len(layer_head) > 1 else None
        button_key = f"search_code{code}_layer{layer}" + (
            f"head{head}" if head is not None else ""
        )
        cols = st.columns(num_search_cols)
        extra_args = {
            "prec": prec,
            "recall": rec,
            "num_acts": code_acts,
            "regex": regex_pattern,
        }
        button_clicked = cols[0].button("🔍", key=button_key)
        if button_clicked:
            webapp_utils.set_ct_acts(code, layer, head, extra_args, is_attn)
        cols[1].write(layer)
        if is_attn:
            cols[2].write(head)
        cols[-4 - non_deploy_offset].write(code)
        cols[-3 - non_deploy_offset].write(f"{prec*100:.2f}%")
        cols[-2 - non_deploy_offset].write(f"{rec*100:.2f}%")
        cols[-1 - non_deploy_offset].write(str(code_acts))
        if not DEPLOY_MODE:
            webapp_utils.add_save_code_button(
                demo_file_path,
                num_acts=code_acts,
                save_regex=True,
                prec=prec,
                recall=rec,
                button_st_container=cols[-1],
                button_key_suffix=f"_code{code}_layer{layer}_head{head}",
            )

    if len(all_codes) == 0:
        st.markdown(
            f"""
            <div style="font-size: 1.0rem; color: red;">
            No codes found for pattern {regex_pattern} at precision threshold: {prec_threshold}
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown("## Code Token Activations")

filter_codes = st.checkbox("Filter Codes", key="filter_codes")
act_range, layer_code_acts = None, None
if filter_codes:
    act_range = st.slider(
        "Num Acts",
        0,
        10_000,
        (100, 10_000),
        key="ct_act_range",
        help="Filter codes by the number of tokens they activate on.",
    )

cols = st.columns(5 if is_attn else 4)
layer = cols[0].number_input("Layer", 0, num_layers - 1, 0, key="ct_act_layer")
if is_attn:
    head = cols[1].number_input("Head", 0, num_heads - 1, 0, key="ct_act_head")
else:
    head = None

def_code = st.session_state.get("ct_act_code", 0)
if filter_codes:
    layer_code_acts = act_count_ft_tkns[
        f"layer{layer}{'_head'+str(head) if head is not None else ''}"
    ]
    def_code = webapp_utils.find_next_code(def_code, layer_code_acts, act_range)
    if "ct_act_code" in st.session_state:
        st.session_state["ct_act_code"] = def_code

code = cols[-3].number_input(
    "Code",
    0,
    num_codes - 1,
    def_code,
    key="ct_act_code",
)
num_examples = cols[-2].number_input(
    "Max Results",
    -1,
    1000,  # setting to 1000 for efficiency purposes even though it can be more than 1000.
    100,
    help="Number of examples to show in the results. Set to -1 to show all examples.",
)
ctx_size = cols[-1].number_input(
    "Context Size",
    1,
    10,
    5,
    help="Number of tokens to show before and after the code token.",
)

acts, acts_count = webapp_utils.get_code_acts(
    model_name,
    tokens_str,
    code,
    layer,
    head,
    ctx_size,
    num_examples,
)

st.write(
    f"Token Activations for Layer {layer}{f' Head {head}' if head is not None else ''} Code {code} | "
    f"Activates on {acts_count[0]} tokens on the acts dataset",
)

if not DEPLOY_MODE:
    webapp_utils.add_save_code_button(
        demo_file_path,
        acts_count[0],
        save_regex=False,
        button_text=True,
        button_key_suffix="_token_acts",
    )

st.markdown(webapp_utils.escape_markdown(acts), unsafe_allow_html=True)
