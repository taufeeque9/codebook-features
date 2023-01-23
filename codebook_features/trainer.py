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

    def log(self, logs: Dict[str, float]) -> None:
        """Adds codebook model related logging.

        Args:
        ----
            logs: log dictionary.
        """
        if isinstance(self.model, models.CodebookModel):
            metric_key_prefix = ""
            if any("train_" in k for k in logs.keys()):
                metric_key_prefix = "train_"
                logs[metric_key_prefix + "learning_rate"] = self._get_learning_rate()
            elif any("eval_" in k for k in logs.keys()):
                metric_key_prefix = "eval_"
            elif any("test_" in k for k in logs.keys()):
                metric_key_prefix = "test_"

            all_codebooks = self.model.all_codebooks
            overall_dead_code_count, dead_code_count, total_codes = 0, 0, 0
            for codebook_idx, codebooks in all_codebooks.items():
                dead_code_count = 0
                for codebook in codebooks:
                    dead_code_count += codebook.num_codes - len(codebook.counts)
                layer_codes = sum(codebook.num_codes for codebook in codebooks)
                logs[metric_key_prefix + f"dead_code_fraction_layer{codebook_idx}"] = (
                    dead_code_count / layer_codes
                )
                overall_dead_code_count += dead_code_count
                total_codes += layer_codes

            logs[metric_key_prefix + "dead_code_fraction"] = (
                overall_dead_code_count / total_codes
            )
        super().log(logs)
