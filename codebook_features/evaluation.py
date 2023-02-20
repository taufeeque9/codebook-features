"""evaluation related functions."""
import dataclasses
from datetime import datetime

import numpy as np
import transformers

from codebook_features import run_clm


def evaluate(model, model_args, data_args, eval_on="train"):
    """Evaluate a model on a dataset."""
    output_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    training_args = transformers.TrainingArguments(output_dir=output_dir)
    eval_args = dataclasses.replace(
        training_args,
        output_dir=training_args.output_dir,
        do_train=False,
        do_eval=True,
        report_to="none",
        per_device_eval_batch_size=32,
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
    dataset = dataset[eval_on]
    tokens = dataset["input_ids"]
    metrics = trainer.evaluate(eval_dataset=dataset)
    for k, v in codebook_acts.items():
        codebook_acts[k] = np.concatenate(v, axis=0)

    np.save(f"{output_dir}/tokens.npy", tokens)
    np.save(f"{output_dir}/cb_acts.npy", codebook_acts)
    np.save(f"{output_dir}/metrics.npy", metrics)

    return tokens, codebook_acts, metrics
