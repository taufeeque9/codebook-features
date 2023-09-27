"""Trainer for codebooks."""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import transformers
import wandb
from torch import nn

from codebook_features import models


class CodebookTrainer(transformers.Trainer):
    """Trainer with additional features for codebook based models."""

    def __init__(
        self,
        model: Union[transformers.PreTrainedModel, nn.Module] = None,
        args: transformers.TrainingArguments = None,
        data_collator: Optional[transformers.DataCollator] = None,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], transformers.PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[transformers.EvalPrediction], Dict]] = None,
        callbacks: Optional[List[transformers.TrainerCallback]] = None,
        optimizers: Tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        """Build the trainer."""
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss for codebook models with regularization."""
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs)

        if isinstance(model, models.CodebookModel) and self.args.codebook_reg_p:
            loss += self.args.codebook_weight_decay * model.codebook_regularization(
                self.args.codebook_reg_p
            )
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """Add codebook model related logging.

        Args:
            logs: log dictionary.
        """
        metric_prefix = ""
        if all("eval_" in k for k in logs.keys()):
            metric_prefix = "eval_"

        if isinstance(self.model, models.CodebookModel):
            logs[metric_prefix + "multicode_k"] = models.BaseSnapFunction.k

            all_codebooks = self.model.all_codebooks
            overall_dead_code_count, dead_code_count, total_codes = 0, 0, 0
            max_norm, mean_norm = 0, 0
            for codebook_idx, codebooks_dict in all_codebooks.items():
                dead_code_count = 0
                for codebook in codebooks_dict.values():
                    dead_code_count += codebook.num_codes - codebook.active_codes
                layer_codes = sum(
                    codebook.num_codes for codebook in codebooks_dict.values()
                )
                if layer_codes:
                    logs[metric_prefix + f"dead_code_fraction/layer{codebook_idx}"] = (
                        dead_code_count / layer_codes
                    )
                    logs[metric_prefix + f"MSE/layer{codebook_idx}"] = sum(
                        codebook.reconstruction_mse
                        for codebook in codebooks_dict.values()
                    ) / len(codebooks_dict)
                    logs[metric_prefix + f"input_norm/layer{codebook_idx}"] = sum(
                        codebook.input_norm for codebook in codebooks_dict.values()
                    ) / len(codebooks_dict)
                    logs[metric_prefix + f"output_norm/layer{codebook_idx}"] = sum(
                        codebook.output_norm for codebook in codebooks_dict.values()
                    ) / len(codebooks_dict)
                    layer_mean_norm = sum(
                        codebook.avg_norm() for codebook in codebooks_dict.values()
                    ) / len(codebooks_dict)
                    if metric_prefix == "eval_":
                        continue
                    layer_max_norm = max(
                        codebook.max_norm() for codebook in codebooks_dict.values()
                    )
                    logs[
                        metric_prefix + f"mean_norm/layer{codebook_idx}"
                    ] = layer_mean_norm
                    logs[
                        metric_prefix + f"max_norm/layer{codebook_idx}"
                    ] = layer_max_norm
                    mean_norm += layer_mean_norm
                    max_norm = max(max_norm, layer_max_norm)
                overall_dead_code_count += dead_code_count
                total_codes += layer_codes

            if total_codes:
                logs[metric_prefix + "dead_code_fraction"] = (
                    overall_dead_code_count / total_codes
                )
                logs[metric_prefix + "MSE"] = sum(
                    logs[metric_prefix + f"MSE/layer{codebook_idx}"]
                    for codebook_idx in all_codebooks
                ) / len(all_codebooks)
                logs[metric_prefix + "input_norm"] = sum(
                    logs[metric_prefix + f"input_norm/layer{codebook_idx}"]
                    for codebook_idx in all_codebooks
                ) / len(all_codebooks)
                logs[metric_prefix + "output_norm"] = sum(
                    logs[metric_prefix + f"output_norm/layer{codebook_idx}"]
                    for codebook_idx in all_codebooks
                ) / len(all_codebooks)
                if metric_prefix != "eval_":
                    logs[metric_prefix + "mean_norm"] = mean_norm / len(all_codebooks)
                    logs[metric_prefix + "max_norm"] = max_norm
        super().log(logs)


class WandbCallback(transformers.integrations.WandbCallback):
    """WandbCallback with codebook model logging."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Add codebook model related logging."""
        metric_prefix = ""
        for k in list(logs.keys()):
            if k.startswith("train_"):
                logs[k[len("train_") :]] = logs.pop(k)
        if all("eval_" in k for k in logs.keys()):
            metric_prefix = "eval_"

        if (
            isinstance(model, models.CodebookModel)
            and model.logging
            and args.local_rank <= 0
        ):
            logs = self.log_code_counts_and_weight_distribution(
                logs, model, metric_prefix
            )

        if args.local_rank <= 0:
            super().on_log(args, state, control, model, logs, **kwargs)

        if (
            isinstance(model, models.CodebookModel)
            and model.logging
            and args.local_rank <= 0
        ):
            for codebook_idx in model.all_codebooks:
                logs.pop(metric_prefix + f"code_counts/layer{codebook_idx}")
                if metric_prefix == "eval_":
                    continue
                logs.pop(metric_prefix + f"code_weights/layer{codebook_idx}")

    def log_code_counts_and_weight_distribution(self, logs, model, metric_prefix):
        """Log the code activation plot and also the weight distribution of the most common codebook feature.

        Running this logging function can be expensive and slow down training. So, it is recommended to run this only
        for debugging purposes.
        """
        for codebook_idx, codebooks_dict in model.all_codebooks.items():
            first_codebook = list(codebooks_dict.values())[0]
            counts = first_codebook.most_common_counts()
            counts = np.stack([np.arange(counts.size), counts], axis=1)
            counts = wandb.Table(
                data=counts,
                columns=["x", "count"],
            )
            logs[metric_prefix + f"code_counts/layer{codebook_idx}"] = wandb.plot_table(
                vega_spec_name="wandb/line/v0",
                data_table=counts,
                fields={"x": "x", "y": "count", "title": "Code Count Distribution"},
            )
            if metric_prefix == "eval_":
                continue
            weight_table = wandb.Table(
                data=first_codebook.get_most_used_code().reshape(-1, 1),
                columns=["weight"],
            )
            logs[
                metric_prefix + f"code_weights/layer{codebook_idx}"
            ] = wandb.plot_table(
                vega_spec_name="interpretability/hist_small_bins",
                data_table=weight_table,
                fields={"value": "weight", "title": "Weight Distribution"},
            )

        model.reset_codebook_metrics()
        return logs


class MultiOptimizer(torch.optim.Optimizer):
    """MultiOptimizer that wraps multiple optimizers."""

    def __init__(self, optimizers):
        """Build the MultiOptimizer."""
        self.optimizers = optimizers

    def step(self, closure=None):
        """Perform a single optimization step."""
        for optimizer in self.optimizers:
            optimizer.step(closure)

    def __setstate__(self, state):
        """Set the state of the all the optimizers."""
        super().__setstate__(state)
        for optimizer in self.optimizers:
            optimizer.__setstate__(state)

    @property
    def param_groups(self):
        """Get the parameter groups of all the optimizers."""
        param_grps = []
        for optimizer in self.optimizers:
            param_grps.extend(optimizer.param_groups)
        return param_grps


class MulticodeKScheduler(transformers.TrainerCallback):
    """K scheduler for multicode models."""

    def __init__(self, k_max, k_min, decay_steps, decay_power=1):
        """Build the K scheduler."""
        self.k_max = k_max
        self.k_min = k_min
        self.decay_steps = decay_steps - 1
        self.decay_power = decay_power

    def k_scheduler(self, step):
        """Get the current K value."""
        return int(
            self.k_max
            - (self.k_max - self.k_min)
            * min(1, (step / self.decay_steps) ** (1 / self.decay_power))
        )

    def on_step_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        """Update the K value."""
        models.BaseSnapFunction.k = self.k_scheduler(state.global_step)

    def on_train_begin(self, *args, **kwargs):
        """Set the K value to the max K."""
        models.BaseSnapFunction.k = self.k_max

    def on_train_end(self, *args, **kwargs):
        """Set the K value to the min K."""
        models.BaseSnapFunction.k = self.k_min
