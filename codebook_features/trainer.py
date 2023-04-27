"""Trainer for codebooks."""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import transformers
from torch import nn

import wandb
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
        model_init: Callable[[], transformers.PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[transformers.EvalPrediction], Dict]] = None,
        callbacks: Optional[List[transformers.TrainerCallback]] = None,
        optimizers: Tuple[
            torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR
        ] = ...,
        preprocess_logits_for_metrics: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
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
        """Adds codebook model related logging.

        Args:
        ----
            logs: log dictionary.
        """
        metric_prefix = ""
        if all("train_" in k for k in logs.keys()):
            metric_prefix = "train_"
        elif all("eval_" in k for k in logs.keys()):
            metric_prefix = "eval_"

        if isinstance(self.model, models.CodebookModel):

            logs[metric_prefix + "multicode_k"] = models.BaseSnapFunction.k

            all_codebooks = self.model.all_codebooks
            overall_dead_code_count, dead_code_count, total_codes = 0, 0, 0
            max_norm, mean_norm = 0, 0
            for codebook_idx, codebooks in all_codebooks.items():
                dead_code_count = 0
                for codebook in codebooks:
                    dead_code_count += codebook.num_codes - codebook.active_codes
                layer_codes = sum(codebook.num_codes for codebook in codebooks)
                if layer_codes:
                    logs[metric_prefix + f"dead_code_fraction/layer{codebook_idx}"] = (
                        dead_code_count / layer_codes
                    )
                    logs[metric_prefix + f"MSE/layer{codebook_idx}"] = sum(
                        codebook.reconstruction_mse for codebook in codebooks
                    ) / len(codebooks)
                    logs[metric_prefix + f"input_norm/layer{codebook_idx}"] = sum(
                        codebook.input_norm for codebook in codebooks
                    ) / len(codebooks)
                    logs[metric_prefix + f"output_norm/layer{codebook_idx}"] = sum(
                        codebook.output_norm for codebook in codebooks
                    ) / len(codebooks)
                    layer_mean_norm = sum(
                        codebook.avg_norm() for codebook in codebooks
                    ) / len(codebooks)
                    if metric_prefix == "eval_":
                        continue
                    layer_max_norm = max(codebook.max_norm() for codebook in codebooks)
                    logs[
                        metric_prefix + f"mean_norm/layer{codebook_idx}"
                    ] = layer_mean_norm
                    logs[
                        metric_prefix + f"max_norm/layer{codebook_idx}"
                    ] = layer_max_norm
                    mean_norm += layer_mean_norm
                    max_norm = max(max_norm, layer_max_norm)
                    # table = wandb.Table(
                    #     data=codebooks[0].most_common_counts(),
                    #     columns=["freq"],
                    # )
                    # logs[
                    #     f"cb_histogram_layer{codebook_idx}"
                    # ] = wandb.plot.histogram(table, "freq", title="Codebook Histogram")
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
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        metric_prefix = ""
        if all("train_" in k for k in logs.keys()):
            metric_prefix = "train_"
        elif all("eval_" in k for k in logs.keys()):
            metric_prefix = "eval_"

        if (
            isinstance(model, models.CodebookModel)
            and control.should_evaluate
            and args.local_rank <= 0
        ):
            for codebook_idx, codebooks in model.all_codebooks.items():
                counts = codebooks[0].most_common_counts()
                counts = np.stack([np.arange(counts.size), counts], axis=1)
                counts = wandb.Table(
                    data=counts,
                    columns=["x", "count"],
                )
                logs[
                    metric_prefix + f"code_counts/layer{codebook_idx}"
                ] = wandb.plot_table(
                    vega_spec_name="wandb/line/v0",
                    data_table=counts,
                    fields={"x": "x", "y": "count", "title": "Code Count Distribution"},
                )
                if metric_prefix == "eval_":
                    continue
                weight_table = wandb.Table(
                    data=codebooks[0].get_most_used_code().reshape(-1, 1),
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

        if control.should_evaluate and args.local_rank <= 0:
            super().on_log(args, state, control, model, logs, **kwargs)

        if (
            isinstance(model, models.CodebookModel)
            and control.should_evaluate
            and args.local_rank <= 0
        ):
            for codebook_idx in model.all_codebooks:
                logs.pop(metric_prefix + f"code_counts/layer{codebook_idx}")
                if metric_prefix == "eval_":
                    continue
                logs.pop(metric_prefix + f"code_weights/layer{codebook_idx}")


class MultiOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self, closure=None):
        for optimizer in self.optimizers:
            optimizer.step(closure)

    def __setstate__(self, state):
        super().__setstate__(state)
        for optimizer in self.optimizers:
            optimizer.__setstate__(state)

    @property
    def param_groups(self):
        param_grps = []
        for optimizer in self.optimizers:
            param_grps.extend(optimizer.param_groups)
        return param_grps


class MulticodeKScheduler(transformers.TrainerCallback):
    def __init__(self, k_max, k_min, decay_steps, decay_power=1):
        self.k_max = k_max
        self.k_min = k_min
        self.decay_steps = decay_steps - 1
        self.decay_power = decay_power

    def k_scheduler(self, step):
        return int(
            self.k_max - (self.k_max - self.k_min) * min(1, (step / self.decay_steps)**(1/self.decay_power))
        )

    def on_step_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        models.BaseSnapFunction.k = self.k_scheduler(state.global_step)

    def on_train_begin(self, *args, **kwargs):
        models.BaseSnapFunction.k = self.k_max

    def on_train_end(self, *args, **kwargs):
        models.BaseSnapFunction.k = self.k_min
