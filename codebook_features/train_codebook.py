"""Train script for Codebook models."""

import dataclasses
import json
import os

import hydra
import omegaconf
import pandas as pd
import torch
import transformers

import wandb
from codebook_features import models, run_clm, trainer

shortened_args = {
    "model_name_or_path": "mod",
    "learning_rate": "lr",
    "per_device_train_batch_size": "bs",
    "codebook_size": "cbs",
    "num_compositional_codebooks": "nccb",
    "layers_to_snap": "cb_layers",
    "similarity_metric": "sim",
    "codebook_at": "cb_at",
    "vqvae_loss": "vqvae",
    "train_model_params": "train_mod",
    "model_lr_factor": "mod_lrf",
    "k_codebook": "k",
    "dataset_name": "ds",
}


@hydra.main(config_path="config", config_name="main")
def main(cfg):
    """Train codebook based models parametrized using hydra.

    Args:
        cfg: hydra config.

    Returns: tuple of metrics for trained model and the baseline metrics.
    """
    training_args = run_clm.TrainingArguments(**cfg.training_args)
    model_args = run_clm.ModelArguments(**cfg.model_args)
    data_args = run_clm.DataTrainingArguments(**cfg.data_args)
    training_args.local_rank = int(os.environ.get("LOCAL_RANK", -1))

    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    # double the batch size for 80 GB GPUs (batch size is set assuming 40 GB GPUs)
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).total_memory / (2**30) > 70:
            training_args.per_device_train_batch_size *= 2
        elif torch.cuda.get_device_properties(0).total_memory / (2**30) > 40:
            training_args.per_device_train_batch_size = (
                int(training_args.per_device_train_batch_size * 1.25) + 1
            )
        cfg_dict["training_args"][
            "per_device_train_batch_size"
        ] = training_args.per_device_train_batch_size
    flat_cfg_dict = pd.json_normalize(cfg_dict, sep="@").to_dict(orient="records")[0]
    flat_cfg_dict = {k.split("@")[-1]: v for k, v in flat_cfg_dict.items()}

    # prepare tags and wandb run name from tags
    tags = sorted(cfg.tags)
    for key in sorted(cfg.tag_keys):
        tags.append(f"{shortened_args[key]}: {flat_cfg_dict[key]}")
    if tags:
        cfg_dict["training_args"]["run_name"] = training_args.run_name = ", ".join(tags)

    if training_args.local_rank <= 0:
        wandb.init(
            project=cfg.project,
            name=training_args.run_name,
            tags=tags,
            settings=wandb.Settings(code_dir="."),
            config=cfg_dict,
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    baseline_output_dir = training_args.output_dir + "_baseline"
    if cfg.get_baseline:
        eval_args = dataclasses.replace(
            training_args,
            output_dir=baseline_output_dir,
            do_train=False,
            do_eval=True,
            report_to="none",
        )
        baseline_metrics = run_clm.main(
            model_args,
            data_args,
            training_args=eval_args,
            model=model,
        )
        baseline_metrics = {"baseline/" + k: v for k, v in baseline_metrics.items()}
        with open(baseline_output_dir + "/metrics.json", "w") as f:
            json.dump(baseline_metrics, f)
    else:
        try:
            with open(baseline_output_dir + "/metrics.json", "r") as f:
                baseline_metrics = json.load(f)
        except FileNotFoundError:
            baseline_metrics = {}
    if training_args.local_rank == 0:
        wandb.log(baseline_metrics, commit=False)
    codebook_config = models.CodebookModelConfig(
        num_codes=cfg.codebook_size,
        num_codebooks=cfg.num_compositional_codebooks,
        layers_to_snap=cfg.layers_to_snap,
        similarity_metric=cfg.similarity_metric,
        codebook_at=cfg.codebook_at,
        vqvae_loss=training_args.vqvae_loss,
        k_codebook=cfg.k_codebook,
    )
    model = models.wrap_codebook(
        model_or_path=model,
        config=codebook_config,
        pretrained_path=cfg.pretrained_path,
    )
    if training_args.train_model_params:
        # model.unfreeze_model_params()
        # params = list(model.parameters())
        params = [
            {
                "params": model.get_codebook_params(),
                "lr": training_args.learning_rate,
                # weight decay for codebook params is used through
                # `codebook_weight_decay` param that is used directly
                # to compute regularized loss.
                "weight_decay": 0.0,
            },
            {
                "params": model.get_model_params(),
                "lr": training_args.model_lr_factor * training_args.learning_rate,
                "weight_decay": training_args.weight_decay,
            },
        ]
    else:
        # model.freeze_model_params()
        params = model.get_codebook_params()
    if len(params) > 0:
        optimizer = torch.optim.AdamW(
            params,
            training_args.learning_rate,
        )
    else:
        RuntimeWarning("Codebook not found in model. Training with model params.")
        optimizer = None

    metrics = run_clm.main(
        model_args,
        data_args,
        training_args=training_args,
        model=model,
        optimizers=(optimizer, None),
        callbacks=[trainer.WandbCallback()],
    )
    return metrics, baseline_metrics


if __name__ == "__main__":
    main()
