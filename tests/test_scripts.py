"""Tests for scripts."""

import hydra
import omegaconf
import pytest
import torch
import transformers

import codebook_features.train_codebook
from codebook_features import models, run_clm


class GradientCheckerOptimizer(torch.optim.AdamW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        for group in self.param_groups:
            for p in group["params"]:
                assert p.grad is not None, f"grad is None for: {p}"
        super().step(*args, **kwargs)

@pytest.mark.parametrize("config_name", ["test", "test_pile"])
def test_train_codebook(config_name):
    with hydra.initialize_config_module(
        version_base=None, config_module="codebook_features.config"
    ):
        # cfg = compose(config_name="test", overrides=["app.user=test_user"])
        cfg = hydra.compose(config_name=config_name)
        ret = codebook_features.train_codebook.main(cfg)
        assert ret is not None


def test_straight_through_gradient_flows():
    training_args = transformers.TrainingArguments(
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
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    model = models.GPT2CodebookModel(model, 2, [-1, -2, -5])
    optimizer = GradientCheckerOptimizer(model.get_codebook_params())
    metrics = run_clm.main(
        model_args,
        data_args,
        training_args=training_args,
        model=model,
        optimizers=(optimizer, None),
    )
    assert metrics is not None


def test_model_saving():
    with hydra.initialize_config_module(
        version_base=None, config_module="codebook_features.config"
    ):
        cfg = hydra.compose(config_name="test")
        with omegaconf.open_dict(cfg):
            cfg.get_baseline = False
            cfg.training_args.save_steps = 1
            cfg.training_args.max_steps = 4
        ret = codebook_features.train_codebook.main(cfg)
        assert ret is not None