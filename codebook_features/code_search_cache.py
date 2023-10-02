"""Script to generate cache for code search webapp."""

import argparse
import glob
import multiprocessing as mp
import os
import pathlib
import pickle
from datetime import datetime
from functools import partial
from typing import Dict

import numpy as np
import transformers

from codebook_features import models, run_clm, utils


def parse_args():
    """Parse command line arguments."""
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
    return args


def store_cb_activations(key, codebook_ids, codebook_acts):
    """Store activations for a given codebook."""
    assert len(codebook_ids.shape) == 3  # (bs, seq_len, k_codebook)
    if key not in codebook_acts:
        codebook_acts[key] = []
    codebook_acts[key].append(codebook_ids)


def get_act_counts_for_comp(
    layer=None, head=None, gcb=False, cb_at=None, cb_acts=None, num_codes=None
):
    """Get the number of activations for each code in the codebook at `layer` and `head`."""
    assert cb_at is not None and cb_acts is not None and num_codes is not None
    comp_name = f"layer{layer}_{cb_at}{f'_gcb{head}' if gcb else ''}"
    ft_tkns = utils.features_to_tokens(comp_name, cb_acts, num_codes=num_codes)
    acts_count = np.array([len(v) for v in ft_tkns])
    return f"layer{layer}{f'_head{head}' if gcb else ''}", acts_count


def save_cache(
    output_dir,
    tokens,
    tokenizer,
    act_count_ft_tkns,
    model_info,
):
    """Save cache to disk."""
    output_dir = pathlib.Path(output_dir)

    tokens_text = tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
    tokens_str = [
        tokenizer.convert_ids_to_tokens(tokens[i]) for i in range(tokens.shape[0])
    ]
    tokens_str = [
        [tokenizer.convert_tokens_to_string([t]) for t in t_list]
        for t_list in tokens_str
    ]
    tokens_str_len_arr = np.array([[len(t) for t in t_list] for t_list in tokens_str])
    token_byte_pos = np.cumsum(tokens_str_len_arr, axis=1)

    np.save(output_dir / "tokens.npy", tokens)
    np.save(output_dir / "tokens_str.npy", tokens_str)
    np.save(output_dir / "tokens_text.npy", tokens_text)
    np.save(output_dir / "token_byte_pos.npy", token_byte_pos)
    with open(output_dir / "act_count_ft_tkns.pkl", "wb") as f:  # type: ignore
        pickle.dump(act_count_ft_tkns, f)

    model_info.save(output_dir / "info.txt")
    print("Saved all data to", str(output_dir))


def main():
    """Generate cache for code search webapp."""
    args = parse_args()
    model_id = args.model_name.split("/")[-1]
    dataset_name = args.dataset_name

    device = "cuda"
    orig_cb_model = models.wrap_codebook(
        model_or_path=args.model_name,
        pretrained_path=args.pretrained_path,
    )
    orig_cb_model = orig_cb_model.to(device).eval()

    assert (
        len(orig_cb_model.config.codebook_type) == 1
    ), "Multi-codebook per layer not supported."

    cb_at = orig_cb_model.config.codebook_at[0]
    is_attn = "attn" in cb_at
    gcb = orig_cb_model.config.codebook_type[0] == "group"
    n_layers = orig_cb_model.num_layers()
    n_heads = orig_cb_model.config.num_codebooks[0] if is_attn else None
    num_codes = orig_cb_model.config.num_codes
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.pretrained_path)

    print("Loaded original cb model.")

    training_args = run_clm.TrainingArguments(
        do_eval=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
    )

    model_args = run_clm.ModelArguments(model_name_or_path=args.pretrained_path)
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
        assert args.dataset_config_name is not None
        data_args = run_clm.DataTrainingArguments(
            dataset_name=dataset_name,
            dataset_config_name=args.dataset_config_name,
            streaming=False,
        )

    trainer, lm_datasets, _, _ = run_clm.get_trainer_and_dataset(
        model_args,
        data_args,
        training_args,
        orig_cb_model,
        optimizers=(None, None),
    )

    # get a random subset of the dataset to cache codebook activations for

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(lm_datasets["train"]), args.max_samples, replace=False)
    dataset = lm_datasets["train"].select(indices)
    tokens = dataset["input_ids"]
    tokens = np.array(tokens)

    cb_acts: Dict[str, np.ndarray] = {}

    # search for output_dir of the form {model_id}_{gcb/vcb}_{cb_at}_{date_time}/
    # if it exists, load the activations from there
    # otherwise, generate the activations and save them to a new output_dir
    # gcb stands for grouped codebook, vcb stands for vanilla codebook
    output_dir = args.output_base_dir + f"{model_id}_{'g' if gcb else 'v'}cb_{cb_at}*"
    dirs = glob.glob(output_dir)
    dirs.sort(key=os.path.getmtime)
    if len(dirs) > 0 and (not args.regen_cache):
        output_dir = dirs[-1]
        print("Loading activations from", output_dir)
        with open(f"{output_dir}/cb_acts.pkl", "rb") as f:
            cb_acts = pickle.load(f)
        tokens = np.load(f"{output_dir}/tokens.npy")
        metrics = np.load(f"{output_dir}/metrics.npy", allow_pickle=True)
    else:
        cb_acts_hook_fn = partial(store_cb_activations, codebook_acts=cb_acts)
        orig_cb_model.set_hook_fn(cb_acts_hook_fn)
        orig_cb_model.reset_codebook_metrics(), orig_cb_model.reset_hook_kwargs()
        orig_cb_model.enable_logging()
        metrics = trainer.evaluate(dataset)
        for k, v in cb_acts.items():
            cb_acts[k] = np.concatenate(v, axis=0)
        output_dir = (
            output_dir[:-1] + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/tokens.npy", tokens)
        np.save(f"{output_dir}/metrics.npy", metrics)
        with open(f"{output_dir}/cb_acts.pkl", "wb") as f:  # type: ignore
            pickle.dump(cb_acts, f)
        print("Saved activations to", output_dir)

    print(metrics)

    # get activation counts for each code in all of the codebooks

    partial_get_act_counts_for_comp = partial(
        get_act_counts_for_comp,
        gcb=gcb,
        cb_at=cb_at,
        cb_acts=cb_acts,
        num_codes=num_codes,
    )
    act_count_ft_tkns = {}
    if is_attn:
        with mp.Pool(min(16, n_heads)) as pool:
            for layer in range(0, n_layers):
                process_component_partial = partial(
                    partial_get_act_counts_for_comp, layer
                )
                for comp_name, acts_count in pool.map(
                    process_component_partial, range(0, n_heads)
                ):
                    act_count_ft_tkns[comp_name] = acts_count
    else:
        with mp.Pool(min(16, n_layers)) as pool:
            for comp_name, acts_count in pool.map(
                partial_get_act_counts_for_comp,
                range(0, n_layers),
            ):
                act_count_ft_tkns[comp_name] = acts_count

    # save cache and model info

    model_info = utils.ModelInfoForWebapp(
        model_name=model_id,
        pretrained_path=args.pretrained_path,
        dataset_name=dataset_name,
        num_codes=num_codes,
        cb_at=cb_at,
        gcb=gcb,
        n_layers=n_layers,
        n_heads=n_heads,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    save_cache(
        output_dir,
        tokens,
        tokenizer,
        act_count_ft_tkns,
        model_info,
    )


if __name__ == "__main__":
    main()
