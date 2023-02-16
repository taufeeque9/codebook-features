"""Trainer for codebooks."""

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import transformers
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
        loss = super().compute_loss(model, inputs, return_outputs)
        if isinstance(model, models.CodebookModel) and self.args.codebook_reg_p:
            loss += self.args.codebook_weight_decay * model.codebook_regularization(
                self.args.codebook_reg_p
            )
        return loss

    def log(self, logs: Dict[str, float]) -> None:
        """Adds codebook model related logging.

        Args:
        ----
            logs: log dictionary.
        """
        if isinstance(self.model, models.CodebookModel):

            all_codebooks = self.model.all_codebooks
            overall_dead_code_count, dead_code_count, total_codes = 0, 0, 0
            max_norm, mean_norm = 0, 0
            for codebook_idx, codebooks in all_codebooks.items():
                dead_code_count = 0
                for codebook in codebooks:
                    dead_code_count += codebook.num_codes - codebook.active_codes
                layer_codes = sum(codebook.num_codes for codebook in codebooks)
                if layer_codes:
                    logs[f"dead_code_fraction/layer{codebook_idx}"] = (
                        dead_code_count / layer_codes
                    )
                    logs[f"MSE/layer{codebook_idx}"] = sum(
                        codebook.reconstruction_mse for codebook in codebooks
                    ) / len(codebooks)
                    layer_mean_norm = sum(
                        codebook.avg_norm() for codebook in codebooks
                    ) / len(codebooks)
                    layer_max_norm = max(codebook.max_norm() for codebook in codebooks)
                    logs[f"mean_norm/layer{codebook_idx}"] = layer_mean_norm
                    logs[f"max_norm/layer{codebook_idx}"] = layer_max_norm
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
                logs["dead_code_fraction"] = overall_dead_code_count / total_codes
                logs["MSE"] = sum(
                    logs[f"MSE/layer{codebook_idx}"] for codebook_idx in all_codebooks
                ) / len(all_codebooks)
                logs["mean_norm"] = mean_norm / len(all_codebooks)
                logs["max_norm"] = max_norm
        super().log(logs)


class WandbCallback(transformers.integrations.WandbCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if isinstance(model, models.CodebookModel):
            # for codebook_idx, codebooks in model.all_codebooks.items():
            #     table = wandb.Table(
            #         data=codebooks[0].most_common_counts(),
            #         columns=["freq"],
            #     )
            #     logs[f"cb_histogram_layer{codebook_idx}"] = wandb.plot.histogram(
            #         table, "freq", title="Codebook Histogram"
            #     )
            model.reset_codebook_metrics()
        super().on_log(args, state, control, model, logs, **kwargs)


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
