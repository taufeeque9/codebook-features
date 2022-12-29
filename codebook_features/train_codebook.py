"""Train script for Codebook models."""

import dataclasses

import hydra
import torch
import transformers

from codebook_features import models, run_clm

# import wandb


@hydra.main(config_path="config", config_name="main")
def main(cfg):
    """Train codebook based models parametrized using hydra.

    Args:
        cfg: hydra config.

    Returns: tuple of metrics for trained model and the baseline metrics.
    """
    training_args = transformers.TrainingArguments(**cfg.training_args)
    model_args = run_clm.ModelArguments(**cfg.model_args)
    data_args = run_clm.DataTrainingArguments(**cfg.data_args)

    model = transformers.GPT2LMHeadModel.from_pretrained(
        cfg.model_args.model_name_or_path,
    )

    if cfg.get_baseline:
        eval_output_dir = training_args.output_dir + "_baseline"
        eval_args = dataclasses.replace(
            training_args,
            output_dir=eval_output_dir,
            do_train=False,
            do_eval=True,
        )
        baseline_metrics = run_clm.main(
            model_args,
            data_args,
            training_args=eval_args,
            model=model,
        )
    else:
        baseline_metrics = None

    model = models.GPT2CodebookModel(model, 100, cfg.layers_to_snap)

    optimizer = torch.optim.AdamW(
        model.get_codebook_params(),
        training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )

    metrics = run_clm.main(
        model_args,
        data_args,
        training_args=training_args,
        model=model,
        optimizers=(optimizer, None),
    )
    return metrics, baseline_metrics


if __name__ == "__main__":
    main()
