"""Web app page for showing codes for different examples in the dataset."""

import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from codebook_features import code_search_utils
from codebook_features.webapp import utils as webapp_utils

webapp_utils.load_widget_state()

if "cb_acts" not in st.session_state:
    switch_page("Code_Browser")

total_examples = 2000
prec_threshold = 0.01

model_name = st.session_state["model_name_id"]
seq_len = st.session_state["seq_len"]
tokens_text = st.session_state["tokens_text"]
tokens_str = st.session_state["tokens_str"]
cb_acts = st.session_state["cb_acts"]
act_count_ft_tkns = st.session_state["act_count_ft_tkns"]
gcb = st.session_state["gcb"]


def get_example_topic_codes(example_id):
    """Get topic codes for the given example id."""
    token_pos_ids = [(example_id, i) for i in range(seq_len)]
    all_codes = []
    for cb_name, cb in cb_acts.items():
        base_cb_name = code_search_utils.convert_to_base_name(cb_name, gcb=gcb)
        codes, prec, rec, code_acts = code_search_utils.get_code_precision_and_recall(
            token_pos_ids,
            cb,
            act_count_ft_tkns[base_cb_name],
        )
        prec_sat_idx = prec >= prec_threshold
        codes, prec, rec, code_acts = (
            codes[prec_sat_idx],
            prec[prec_sat_idx],
            rec[prec_sat_idx],
            code_acts[prec_sat_idx],
        )
        rec_sat_idx = rec >= recall_threshold
        codes, prec, rec, code_acts = (
            codes[rec_sat_idx],
            prec[rec_sat_idx],
            rec[rec_sat_idx],
            code_acts[rec_sat_idx],
        )
        codes_pr = list(zip(codes, prec, rec, code_acts))
        all_codes.append((cb_name, codes_pr))
    return all_codes


def find_next_example(example_id):
    """Find the example after `example_id` that has topic codes."""
    initial_example_id = example_id
    example_id += 1
    while example_id != initial_example_id:
        all_codes = get_example_topic_codes(example_id)
        codes_found = sum([len(code_pr_infos) for _, code_pr_infos in all_codes])
        if codes_found > 0:
            st.session_state["example_id"] = example_id
            return
        example_id = (example_id + 1) % total_examples
    st.error(
        f"No examples found at the specified recall threshold: {recall_threshold}.",
        icon="🚨",
    )


def redirect_to_main_with_code(code, layer, head):
    """Redirect to main page with the given code."""
    st.session_state["ct_act_code"] = code
    st.session_state["ct_act_layer"] = layer
    if st.session_state["multiple_cbs"]:
        st.session_state["ct_act_head"] = head
    switch_page("Code Browser")


def show_examples_for_topic_code(code, layer, head, code_act_ratio=0.3):
    """Show examples that the code activates on."""
    ex_acts, _ = webapp_utils.get_code_acts(
        model_name,
        tokens_str,
        code,
        layer,
        head,
        ctx_size=5,
        return_example_list=True,
    )
    filt_ex_acts = []
    for act_str, num_acts in ex_acts:
        if num_acts > seq_len * code_act_ratio:
            filt_ex_acts.append(act_str)
    st.markdown("#### Examples for Code")
    st.markdown(webapp_utils.escape_markdown("".join(filt_ex_acts)), unsafe_allow_html=True)


multiple_cbs = st.session_state["multiple_cbs"]

st.markdown("## Topic Code")
topic_code_description = (
    "Topic codes are codes that activate many different times on passages that describe a particular"
    " topic or concept (e.g. “fire”). This interface provides a way to search for such codes by looking"
    " at different examples in the dataset (ExampleID) and finding codes that activate on some fraction"
    " of the tokens in that example (Recall Threshold). Decrease the Recall Threshold to view more possible"
    " topic codes and increase it to see fewer. Click “Find Next Example” to find the next example with at"
    " least one code firing on that example above the Recall Threshold."
)
st.write(topic_code_description)

ex_col, r_col, trunc_col, sort_col = st.columns([1, 1, 1, 1])
example_id = ex_col.number_input(
    "Example ID",
    0,
    total_examples - 1,
    0,
    key="example_id",
)
recall_threshold = r_col.slider(
    "Recall Threshold",
    0.0,
    1.0,
    0.2,
    key="recall",
    help="Recall Threshold is the minimum fraction of tokens in the example that the code must activate on.",
)
example_truncation = trunc_col.number_input("Max Output Chars", 0, 102400, 1024, key="max_chars")
sort_by_options = ["Precision", "Recall", "Num Acts"]
sort_by_name = sort_col.radio(
    "Sort By",
    sort_by_options,
    index=1,
    horizontal=True,
    help="Sorts the codes by the selected metric.",
)
sort_by = sort_by_options.index(sort_by_name)


button = st.button(
    "Find Next Example",
    key="find_next_example",
    on_click=find_next_example,
    args=(example_id,),
    help="Find an example which has codes above the recall threshold.",
)

st.markdown("### Example Text")
trunc_suffix = "..." if example_truncation < len(tokens_text[example_id]) else ""
st.write(tokens_text[example_id][:example_truncation] + trunc_suffix)

cols = st.columns(7 if multiple_cbs else 6)
cols[0].markdown("Search", help="Button to see token activations for the code.")
cols[1].write("Layer")
if multiple_cbs:
    cols[2].write("Head")
cols[-4].write("Code")
cols[-3].write("Precision")
cols[-2].write("Recall")
cols[-1].markdown(
    "Num Acts",
    help="Number of tokens that the code activates on in the acts dataset.",
)

all_codes = get_example_topic_codes(example_id)
all_codes = [(cb_name, code_pr_info) for cb_name, code_pr_infos in all_codes for code_pr_info in code_pr_infos]
all_codes = sorted(all_codes, key=lambda x: x[1][1 + sort_by], reverse=True)

for cb_name, (code, p, r, acts) in all_codes:
    cols = st.columns(7 if multiple_cbs else 6)
    code_button = cols[0].button(
        "🔍",
        key=f"ex-code-{code}-{cb_name}",
    )
    layer, head = code_search_utils.get_layer_head_from_adv_name(cb_name)
    cols[1].write(str(layer))
    if multiple_cbs:
        cols[2].write(str(head))

    cols[-4].write(code)
    cols[-3].write(f"{p*100:.2f}%")
    cols[-2].write(f"{r*100:.2f}%")
    cols[-1].write(str(acts))

    if code_button:
        show_examples_for_topic_code(
            code,
            layer,
            head,
            code_act_ratio=recall_threshold,
        )
if len(all_codes) == 0:
    st.markdown(
        f"<div style='text-align:center'>No codes found at recall threshold = {recall_threshold}."
        " Consider decreasing the recall threshold.</div>",
        unsafe_allow_html=True,
    )
