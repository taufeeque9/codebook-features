"""Script to generate cache for code search webapp."""

import argparse
import glob
import multiprocessing as mp
import os
import pickle
from datetime import datetime
from functools import partial

import numpy as np
import transformers

from codebook_features import models, run_clm, utils

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--pretrained_path", type=str, required=True)
parser.add_argument("--max_samples", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_base_dir", type=str, default="/shared/cb_eval_acts/")
parser.add_argument("--regen_cache", default=False, action="store_true")
parser.add_argument("--dataset_name", type=str, default=None)
parser.add_argument("--dataset_config_name", type=str, default=None)


args = parser.parse_args()
model_name_or_path = args.model_name
model_id = model_name_or_path.split("/")[-1]
pretrained_path = args.pretrained_path
max_samples = args.max_samples
seed = args.seed
base_dir = args.output_base_dir
regen_cache = args.regen_cache
dataset_name = args.dataset_name
dataset_config_name = args.dataset_config_name

device = "cuda"
orig_cb_model = models.wrap_codebook(
    model_or_path=model_name_or_path, pretrained_path=pretrained_path
)
orig_cb_model = orig_cb_model.to(device).eval()
orig_cb_model.disable_logging()

assert (
    len(orig_cb_model.config.codebook_type) == 1
), "Multi-codebook per layer not supported."

cb_at_dict = {
    "preproj_attention": "attn_preproj",
    "transformer_block": "tb",
    "attention": "attn",
    "attention_and_mlp": "attn+mlp",
}

cb_at = orig_cb_model.config.codebook_at[0]
cb_at = cb_at_dict.get(cb_at, cb_at)
is_attn = "attn" in cb_at
ccb = orig_cb_model.config.codebook_type[0] == "compositional"
n_layers = orig_cb_model.num_layers()
n_heads = orig_cb_model.config.num_codebooks[0] if is_attn else None
num_codes = orig_cb_model.config.num_codes
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_path)

print("Loaded original cb model.")

report_to = "none"
# report_to = "all"
training_args = run_clm.TrainingArguments(
    #     no_cuda=True,
    output_dir=f"/shared/output_{cb_at}_{model_id}/",
    do_eval=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
)

model_args = run_clm.ModelArguments(
    model_name_or_path=pretrained_path, cache_dir="/shared/.cache/huggingface/"
)
if not dataset_name:
    if "tinystories" in model_id.lower():
        dataset_name = "roneneldan/TinyStories"
        data_args = run_clm.DataTrainingArguments(
            dataset_name=dataset_name,
            dataset_config_name=None,
            streaming=False,
            block_size=512,
        )
    else:
        dataset_name = "wikitext"
        data_args = run_clm.DataTrainingArguments(
            dataset_name=dataset_name,
            dataset_config_name="wikitext-103-v1",
            streaming=False,
        )
else:  # used when tinystories is trained on wikitext for example
    assert dataset_config_name is not None
    data_args = run_clm.DataTrainingArguments(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        streaming=False,
    )

trainer, lm_datasets, raw_datasets, last_checkpoint = run_clm.get_trainer_and_dataset(
    model_args,
    data_args,
    training_args,
    orig_cb_model,
    optimizers=(None, None),
)

rng = np.random.default_rng(seed)
indices = rng.choice(len(lm_datasets["train"]), max_samples, replace=False)
dataset = lm_datasets["train"].select(indices)
tokens = dataset["input_ids"]
tokens = np.array(tokens)
tokens_text = tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)

cb_acts = {}
ft_tkns = {}


def store_cb_activations(key, codebook_ids, codebook_acts=cb_acts):
    """Store activations for a given codebook."""
    assert len(codebook_ids.shape) == 3  # (bs, seq_len, k_codebook)
    if key not in codebook_acts:
        codebook_acts[key] = []
    codebook_acts[key].append(codebook_ids)


output_dir = base_dir + f"{model_id}_{'c' if ccb else 'v'}cb_{cb_at}*"
dirs = glob.glob(output_dir)
dirs.sort(key=os.path.getmtime)
if len(dirs) > 0 and (not regen_cache):
    output_dir = dirs[-1]
    print("Loading activations from", output_dir)
    with open(f"{output_dir}/cb_acts.pkl", "rb") as f:
        cb_acts = pickle.load(f)
    tokens = np.load(f"{output_dir}/tokens.npy")
    metrics = np.load(f"{output_dir}/metrics.npy", allow_pickle=True)
else:
    orig_cb_model.set_hook_fn(store_cb_activations)
    orig_cb_model.reset_codebook_metrics(), orig_cb_model.reset_hook_kwargs()
    orig_cb_model.enable_logging()
    metrics = trainer.evaluate(dataset)
    for k, v in cb_acts.items():
        cb_acts[k] = np.concatenate(v, axis=0)
    output_dir = output_dir[:-1] + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/tokens.npy", tokens)
    np.save(f"{output_dir}/metrics.npy", metrics)
    with open(f"{output_dir}/cb_acts.pkl", "wb") as f:
        pickle.dump(cb_acts, f)
    print("Saved activations to", output_dir)

print(metrics)

tokens_str = [
    tokenizer.convert_ids_to_tokens(tokens[i]) for i in range(tokens.shape[0])
]
tokens_str = [
    [tokenizer.convert_tokens_to_string([t]) for t in t_list] for t_list in tokens_str
]
tokens_str_len_arr = np.array([[len(t) for t in t_list] for t_list in tokens_str])
token_byte_pos = np.cumsum(tokens_str_len_arr, axis=1)


def get_act_counts_for_comp(layer=None, head=None):
    """Get the number of activations for each code in the codebook at `layer` and `head`."""
    comp_name = f"layer{layer}_{cb_at}{f'_ccb{head}' if ccb else ''}"
    ft_tkns = utils.features_to_tokens(comp_name, cb_acts, num_codes=num_codes)
    acts_count = np.array([len(v) for v in ft_tkns])
    return f"layer{layer}{f'_head{head}' if ccb else ''}", acts_count


act_count_ft_tkns = {}
if is_attn:
    with mp.Pool(min(16, n_heads)) as pool:
        for layer in range(0, n_layers):
            process_component_partial = partial(get_act_counts_for_comp, layer)
            for comp_name, acts_count in pool.map(
                process_component_partial, range(0, n_heads)
            ):
                act_count_ft_tkns[comp_name] = acts_count
else:
    with mp.Pool(min(16, n_layers)) as pool:
        for comp_name, acts_count in pool.map(
            get_act_counts_for_comp, range(0, n_layers)
        ):
            act_count_ft_tkns[comp_name] = acts_count

# save all the data
np.save(output_dir + "/tokens.npy", tokens)
np.save(output_dir + "/tokens_str.npy", tokens_str)
np.save(output_dir + "/tokens_text.npy", tokens_text)
np.save(output_dir + "/token_byte_pos.npy", token_byte_pos)
with open(output_dir + "/act_count_ft_tkns.pkl", "wb") as f:
    pickle.dump(act_count_ft_tkns, f)

with open(output_dir + "/info.txt", "w") as f:
    f.write(f"num_codes: {num_codes}\n")
    f.write(f"cb_at: {cb_at}\n")
    f.write(f"ccb: {ccb}\n")
    f.write(f"n_layers: {n_layers}\n")
    f.write(f"n_heads: {n_heads}\n")
    f.write(f"model_name: {model_name_or_path}\n")
    f.write(f"pretrained_path: {pretrained_path}\n")
    f.write(f"seed: {seed}\n")
    f.write(f"max_samples: {max_samples}\n")
    f.write(f"dataset_name: {dataset_name}\n")

print("Saved all data to", output_dir)
