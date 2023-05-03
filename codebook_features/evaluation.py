"""evaluation related functions."""
import dataclasses
import pickle
from datetime import datetime

import numpy as np

from codebook_features import models, run_clm


def evaluate(model, model_args, data_args, eval_on="train", save_dir=None):
    """Evaluate a model on a dataset."""
    output_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if save_dir is not None:
        output_dir = f"{save_dir}/{output_dir}"

    eval_args = run_clm.TrainingArguments(
        output_dir=output_dir,
        tf32=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=64,
        do_train=True,
        do_eval=True,
        report_to="none",
    )
    codebook_acts = {}

    def store_cb_activations(key, codebook_ids, codebook_acts=codebook_acts):
        assert len(codebook_ids.shape) == 3  # (bs, seq_len, k_codebook)
        if key not in codebook_acts:
            codebook_acts[key] = []
        codebook_acts[key].append(codebook_ids)

    model.set_hook_fn(store_cb_activations)
    trainer, dataset, _ = run_clm.get_trainer_and_dataset(
        model_args,
        data_args,
        training_args=eval_args,
        model=model,
    )
    dataset = trainer.train_dataset if eval_on == "train" else trainer.eval_dataset
    tokens = dataset["input_ids"]
    print("tokens shape:", len(tokens))
    if isinstance(model, models.HookedTransformerCodebookModel):
        dataset.rename_column("input_ids", "input")

    print("************Starting evaluation************")
    metrics = trainer.evaluate(eval_dataset=dataset)
    for k, v in codebook_acts.items():
        codebook_acts[k] = np.concatenate(v, axis=0)

    np.save(f"{output_dir}/tokens.npy", tokens)
    with open(f"{output_dir}/cb_acts.pkl", "wb") as f:
        pickle.dump(codebook_acts, f)
    np.save(f"{output_dir}/metrics.npy", metrics)

    return tokens, codebook_acts, metrics, output_dir
