"""Tests for scripts."""

import hydra
import omegaconf
import pytest
import torch
import transformers

from codebook_features import models, run_clm, train_codebook, train_toy_model


class GradientCheckerOptimizer(torch.optim.AdamW):
    """Optimizer that checks if all of the gradients are real."""

    def step(self, *args, **kwargs):
        """Check if all of the gradients are real and step."""
        for group in self.param_groups:
            for p in group["params"]:
                assert p.grad is not None, f"grad is None for: {p}"
        super().step(*args, **kwargs)


@pytest.mark.parametrize(
    "config_name", ["test", "test_pile", "test_pythia", "test_vqtorch"]
)
def test_train_codebook(config_name):
    """Test training codebook script."""
    with hydra.initialize_config_module(
        version_base=None, config_module="codebook_features.config"
    ):
        cfg = hydra.compose(config_name=config_name)
        ret = train_codebook.main(cfg)
        assert ret is not None


def test_train_toy_codebook():
    """Test training toy codebook script."""
    with hydra.initialize_config_module(
        version_base=None, config_module="codebook_features.config"
    ):
        cfg = hydra.compose(config_name="test_toy")
        ret = train_toy_model.main(cfg)
        assert ret is not None


def test_straight_through_gradient_flows():
    """Test that gradients flow through the codebook."""
    training_args = run_clm.TrainingArguments(
        "test_output",
        overwrite_output_dir=True,
        max_steps=1,
        do_train=True,
    )
    model_args = run_clm.ModelArguments(model_name_or_path="gpt2")
    data_args = run_clm.DataTrainingArguments(
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        max_train_samples=2,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained("taufeeque/tiny-gpt2")
    cb_config = models.CodebookModelConfig(
        num_codes=10, k_codebook=2, layers_to_snap="all"
    )
    model = models.GPT2CodebookModel(model=model, config=cb_config)
    optimizer = GradientCheckerOptimizer(model.get_codebook_params())
    trainer, lm_dataset, last_checkpoint = run_clm.get_trainer_and_dataset(
        model_args, data_args, training_args, model, optimizers=(optimizer, None)
    )
    metrics = run_clm.run_trainer(
        model_args, data_args, training_args, trainer, lm_dataset, last_checkpoint
    )
    assert metrics is not None


def test_model_saving():
    """Test that model is saved without any errors."""
    with hydra.initialize_config_module(
        version_base=None, config_module="codebook_features.config"
    ):
        cfg = hydra.compose(config_name="test")
        with omegaconf.open_dict(cfg):
            cfg.get_baseline = False
            cfg.training_args.save_steps = 1
            cfg.training_args.max_steps = 2
        ret = train_codebook.main(cfg)
        assert ret is not None
