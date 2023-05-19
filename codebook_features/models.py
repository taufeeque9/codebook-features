"""Model related classes."""

import abc
import os
import re
from collections import Counter
from typing import Any, Callable, Dict, Optional, Sequence, Type, Union

import numpy as np
import torch
import transformer_lens
import transformers
from sklearn import cluster as sklearn_cluster
from torch import nn
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from codebook_features import mod_model_classes


class KmeansEmbedding(nn.Embedding):
    """Embedding module that can be initialized to the kmeans of batched data."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
        **kmeans_kwargs,
    ):
        """Create K-Means Embedding.

        Args:
        ----
            num_embeddings: size of the dictionary of embeddings
            embedding_dim: the size of each embedding vector
            padding_idx: padding index. Defaults to None.
            max_norm: If given, each embedding vector with norm larger than :attr:`max_norm`
                is renormalized to have norm :attr:`max_norm`. Defaults to None.
            norm_type: The p of the p-norm to compute for the :attr:`max_norm` option. Defaults to 2.0.
            scale_grad_by_freq: If given, this will scale gradients by the inverse of frequency of
                the words in the mini-batch. Defaults to False.
            sparse: If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor. Defaults to False.
            _weight: the learnable weights of the module. Defaults to None.
            device: device to put torch tensors on. Defaults to None.
            dtype: data type of embedding. Defaults to None.
        """
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            device,
            dtype,
        )
        self.data = None
        self.kmeans = sklearn_cluster.MiniBatchKMeans(
            n_clusters=num_embeddings, **kmeans_kwargs
        )

    def load_data(self, data: torch.Tensor):
        """Load a batch of data.

        Args:
        ----
            data: batch of data.
        """
        with torch.no_grad():
            if self.data is None:
                self.data = data
            else:
                self.data = torch.cat([self.data, data], dim=0)

    def clear_data(self):
        """Clear the data."""
        self.data = None

    def partial_fit(self):
        """Fit the K-Means model to the loaded data."""
        self.data = self.data.reshape(-1, self.embedding_dim)
        self.kmeans.partial_fit(self.data.detach().cpu().numpy())
        self.clear_data()

    def initialize(self):
        """Initialize the embeddings after all the data is loaded.

        Args:
        ----
            k: number of cluster centers for K-Means.
        """
        self.weight.data = torch.from_numpy(self.kmeans.cluster_centers_).to(
            self.weight.device
        )
        self.clear_data()


class BaseSnapFunction(torch.autograd.Function):
    """Autograd Fn to snap input to closest codebook feature.
    This is the base class. It should be subclassed with a forward function."""

    loss = "base"
    k = 1

    @staticmethod
    def backward(ctx, grad_outputs, grad_codebook_ids):
        """Backward pass for the snap function using straight-through operator.

        Args:
        ----
            ctx: torch context used for efficiently storing tensors for backward pass.
            grad_outputs: gradient tensor of the outputs.
            grad_codebook_ids: gradient tensor of `codebook_ids`.

        Returns: tuple of gradient tensor wrt `inputs` and `codebook` tensors.
        """
        inputs, codebook, outputs = ctx.saved_tensors
        if BaseSnapFunction.loss[:5] == "vqvae":
            try:
                beta = float(BaseSnapFunction.loss.split("-")[1])
            except IndexError:
                beta = 0.25
            with torch.enable_grad():
                mse_loss = torch.mean(((outputs - inputs) ** 2).sum(dim=-1))
                grad_codebook = torch.autograd.grad(mse_loss, codebook)[0]
            # straight through estimator + commitment loss gradient
            grad_inputs = grad_outputs + (2 * beta) * (inputs - outputs)
        elif BaseSnapFunction.loss == "base":
            grad_codebook = torch.autograd.grad(outputs, codebook, grad_outputs)[0]
            # straight through estimator
            grad_inputs = grad_outputs
        elif BaseSnapFunction.loss == "aeloss":
            # grad_codebook_mse = torch.autograd.grad(mse, codebook, retain_graph=True)[0]
            with torch.enable_grad():
                mse_loss = torch.mean(((outputs - inputs) ** 2).sum(dim=-1))
                grad_codebook_mse = torch.autograd.grad(
                    mse_loss, codebook, retain_graph=True
                )[0]
            grad_codebook = torch.autograd.grad(outputs, codebook, grad_outputs)[0]

            grad_codebook += grad_codebook_mse
            # straight through estimator
            grad_inputs = grad_outputs
        elif BaseSnapFunction.loss[:10] == "fullaeloss":
            try:
                beta = float(BaseSnapFunction.loss.split("-")[1])
            except IndexError:
                beta = 0.25
            with torch.enable_grad():
                mse_loss = torch.mean(((outputs - inputs) ** 2).sum(dim=-1))
                grad_codebook_mse = torch.autograd.grad(
                    mse_loss, codebook, retain_graph=True
                )[0]
            grad_codebook = torch.autograd.grad(outputs, codebook, grad_outputs)[0]

            grad_codebook += grad_codebook_mse
            # straight through estimator
            grad_inputs = grad_outputs + 2 * beta * (inputs - outputs)

        return grad_inputs, grad_codebook, None, None


class InnerProductSnapFunction(BaseSnapFunction):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, codebook: torch.Tensor, kcodes, hook_kwargs):
        """Compute output of the snap function with the maximum inner product
        as the similarity metric.

        Replaces each dimension vector of input with features from codebook
        having highest dot-product.

        Args:
        ----
            ctx: torch context used for efficiently storing tensors for backward pass.
            inputs: input data.
            codebook: codebook matrix. Shape: (num_features, hidden_dim_size).

        Returns: tuple of output of snap function and the IDs of closest codebook features.
        """
        logits = torch.matmul(inputs, codebook.T)
        if hook_kwargs["cosine"]:
            logits = logits / (
                torch.norm(inputs, dim=-1, keepdim=True)
                * torch.norm(codebook, dim=-1, keepdim=True).T
            )
        if hook_kwargs["keep_k_codes"]:
            if hook_kwargs["disable_for_tkns"] == "all":
                logits[:, :, hook_kwargs["disable_codes"]] = float("-inf")
                _, codebook_ids = logits.topk(
                    kcodes + hook_kwargs["disable_topk"], dim=-1
                )
                codebook_ids = codebook_ids[:, :, -kcodes :]
            else:
                for code in hook_kwargs["disable_codes"]:
                    logits[:, hook_kwargs["disable_for_tkns"], code] = float("-inf")
                _, codebook_ids_all = logits.topk(
                    kcodes + hook_kwargs["disable_topk"], dim=-1
                )
                codebook_ids = codebook_ids_all[:, :, : kcodes]
                codebook_ids[:, hook_kwargs["disable_for_tkns"]] = codebook_ids_all[
                    :, hook_kwargs["disable_for_tkns"], -kcodes :
                ]
            # enable gradient so that outputs.grad_fn can be used in backward pass.
            with torch.enable_grad():
                outputs = torch.nn.functional.embedding(codebook_ids, codebook)
                outputs = outputs.sum(dim=-2) / kcodes
        else:
            _, codebook_ids = logits.topk(kcodes, dim=-1)
            input_codes = torch.nn.functional.embedding(codebook_ids, codebook)
            if hook_kwargs["disable_for_tkns"] == "all":
                unblock_tkns = torch.zeros(
                    codebook_ids.shape[1], dtype=torch.bool, device=codebook_ids.device
                )
            else:
                unblock_tkns = torch.ones(
                    codebook_ids.shape[1], dtype=torch.bool, device=codebook_ids.device
                )
                unblock_tkns[hook_kwargs["disable_for_tkns"]] = False
            block_idx = torch.isin(
                codebook_ids, torch.tensor(hook_kwargs["disable_codes"], device=codebook_ids.device)
            )
            block_idx[:, :, : hook_kwargs["disable_topk"]] = True
            block_idx[:, unblock_tkns, :] = False
            block_idx[:, :, hook_kwargs["disable_sim_idx"]] = True
            codebook_ids[block_idx] = -1
            input_codes[block_idx] = 0
            outputs = input_codes.sum(dim=-2) / kcodes

        ctx.save_for_backward(inputs, codebook, outputs)
        # detach & clone outputs since the returned tensor's grad_fn will be
        # overridden by SnapFunction.backward and we don't want the above
        # outputs.grad_fn to be overridden.
        return outputs.detach().clone(), codebook_ids


class EuclideanSnapFunction(BaseSnapFunction):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, codebook: torch.Tensor, kcodes, hook_kwargs):
        """Compute output of the snap function with the minimum euclidean
        distance as the similarity metric.

        Replaces each dimension vector of input with features from codebook
        having highest dot-product.

        Args:
        ----
            ctx: torch context used for efficiently storing tensors for backward pass.
            inputs: input data.
            codebook: codebook matrix. Shape: (num_features, hidden_dim_size).

        Returns: tuple of output of snap function and the IDs of closest codebook features.
        """
        logits = -torch.cdist(inputs, codebook, p=2)  # logits are negative distances
        if hook_kwargs["keep_k_codes"]:
            if hook_kwargs["disable_for_tkns"] == "all":
                logits[:, :, hook_kwargs["disable_codes"]] = float("-inf")
                _, codebook_ids = logits.topk(
                    kcodes + hook_kwargs["disable_topk"], dim=-1
                )
                codebook_ids = codebook_ids[:, :, -kcodes :]
            else:
                for code in hook_kwargs["disable_codes"]:
                    logits[:, hook_kwargs["disable_for_tkns"], code] = float("-inf")
                _, codebook_ids_all = logits.topk(
                    kcodes + hook_kwargs["disable_topk"], dim=-1
                )
                codebook_ids = codebook_ids_all[:, :, : kcodes]
                codebook_ids[:, hook_kwargs["disable_for_tkns"]] = codebook_ids_all[
                    :, hook_kwargs["disable_for_tkns"], -kcodes :
                ]
            # enable gradient so that outputs.grad_fn can be used in backward pass.
            with torch.enable_grad():
                outputs = torch.nn.functional.embedding(codebook_ids, codebook)
                outputs = outputs.sum(dim=-2) / kcodes
        else:
            _, codebook_ids = logits.topk(kcodes, dim=-1)
            input_codes = torch.nn.functional.embedding(codebook_ids, codebook)
            if hook_kwargs["disable_for_tkns"] == "all":
                unblock_tkns = torch.zeros(
                    codebook_ids.shape[1], dtype=torch.bool, device=codebook_ids.device
                )
            else:
                unblock_tkns = torch.ones(
                    codebook_ids.shape[1], dtype=torch.bool, device=codebook_ids.device
                )
                unblock_tkns[hook_kwargs["disable_for_tkns"]] = False
            block_idx = torch.isin(
                codebook_ids, torch.tensor(hook_kwargs["disable_codes"], device=codebook_ids.device)
            )
            block_idx[:, :, : hook_kwargs["disable_topk"]] = True
            block_idx[:, unblock_tkns, :] = False
            codebook_ids[block_idx] = -1
            input_codes[block_idx] = 0
            outputs = input_codes.sum(dim=-2) / kcodes

        ctx.save_for_backward(inputs, codebook, outputs)
        # detach & clone outputs since the returned tensor's grad_fn will be
        # overridden by SnapFunction.backward and we don't want the above
        # outputs.grad_fn to be overridden.
        return outputs.detach().clone(), codebook_ids


class CompostionalEuclideanSnapFunction(BaseSnapFunction):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, codebook: torch.Tensor):
        """Compute output of the snap function with the minimum euclidean
        distance as the similarity metric.

        Replaces each dimension vector of input with features from codebook
        having highest dot-product.

        Args:
        ----
            ctx: torch context used for efficiently storing tensors for backward pass.
            inputs: input data.
            codebook: codebook matrix. Shape: (num_features, hidden_dim_size).

        Returns: tuple of output of snap function and the IDs of closest codebook features.
        """
        comp_input = inputs.reshape(
            inputs.shape[0], inputs.shape[1], codebook.shape[0], -1
        ).permute(0, 2, 1, 3)
        logits = -torch.cdist(comp_input, codebook, p=2).permute(
            0, 2, 1, 3
        )  # logits are negative distances
        # logits has shape (bs, seq_len, #ccb, cb_size)
        codebook_ids = logits.topk(BaseSnapFunction.k, dim=-1)[1]
        # enable gradient so that outputs.grad_fn can be used in backward pass.
        with torch.enable_grad():
            outputs = torch.concat(
                [
                    torch.nn.functional.embedding(codebook_ids[:, :, i, :], codebook[i])
                    for i in range(codebook.shape[0])
                ],
                dim=-1,
            )
            outputs = outputs.sum(dim=-2) / BaseSnapFunction.k
        # ctx.save_for_backward(codebook, outputs)
        ctx.save_for_backward(inputs, codebook, codebook_ids, outputs)
        # detach & clone outputs since the returned tensor's grad_fn will be
        # overridden by SnapFunction.backward and we don't want the above
        # outputs.grad_fn to be overridden.
        return outputs.detach().clone(), codebook_ids


class CodebookLayer(nn.Module):
    """Codebook layer module."""

    def __init__(
        self,
        dim: int,
        num_codes: int,
        key: str,
        soft_snap: bool = False,
        snap_fn: BaseSnapFunction = EuclideanSnapFunction,
        hook_fn: Callable = None,
        kmeans_init=False,
        kmeans_kwargs: dict = {},
        kcodes=1,
        **kwargs,
    ):
        """Create the codebook layer.

        Args:
        ----
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            kmeans_init: whether to initialize the codebook with k-means of the data. Defaults to False.
            soft_snap: whether to snap the input using softmax. Defaults to False.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
        """
        super().__init__()
        if kmeans_init:
            self.codebook = KmeansEmbedding(
                num_embeddings=num_codes, embedding_dim=dim, **kmeans_kwargs
            )
        else:
            self.codebook = nn.Embedding(num_embeddings=num_codes, embedding_dim=dim)
        self.ln = torch.nn.LayerNorm(dim, eps=1e-05)
        self._num_codes = num_codes
        self.counts = torch.zeros(num_codes, dtype=torch.long)
        self.soft_snap = soft_snap
        self.snap_fn = snap_fn
        self.hook_fn = hook_fn
        self.key = key
        self.kcodes = kcodes
        self.reconstruction_mse = 0
        self.input_norm = 0
        self.output_norm = 0
        self.tokens_processed = 0
        self.reset_hook_kwargs()
        self.hook_codebook_ids = HookPoint()
        self.logging = True

    @property
    def active_codes(self):
        """Return the number of active codes."""
        return torch.sum(self.counts != 0).item()

    @property
    def num_codes(self):
        """Return the total number of codes."""
        return self._num_codes

    def enable_logging(self):
        self.logging = True

    def disable_logging(self):
        self.logging = False

    def initialize_codebook(self, data: torch.Tensor):
        """Initialize the codebook using k-means.

        Args:
        ----
            data: data to use for initializing the codebook.
        """
        assert isinstance(self.codebook, KmeansEmbedding)
        self.codebook.initialize(data=data)

    def get_most_used_code(self):
        """Return the most used code."""
        return self.codebook.weight[self.counts.argmax()].detach().cpu().numpy()

    def set_hook_fn(self, hook_fn: Callable):
        """Set the hook function.

        Args:
        ----
            hook_fn: hook function to use.
        """
        self.hook_fn = hook_fn

    def forward(self, x: torch.Tensor):
        """Snaps activations to elements in the codebook.

        Args:
        ----
            x: input tensor of shape: (batch_size, seq_len, dim).

        Returns: output with the feature vectors replaced using the codebook.
        """
        assert len(x.shape) == 3  # (batch_size, seq_len, dim)
        if not self.soft_snap:
            # Hard choice of a single codebook vector
            output, codebook_ids = self.snap_fn.apply(
                self.ln(x), self.codebook.weight, self.kcodes, self.hook_kwargs,
            )
            codebook_ids = self.hook_codebook_ids(codebook_ids)
            if not self.training:
                block_idx = torch.isin(codebook_ids, -1)
                # TODO: clean this.
                # change -1 to 0 and zero ablate the blocked codes before computing mean
                codebook_ids[block_idx] = 0
                output = self.codebook(codebook_ids)
                output[block_idx] = 0
                output = output.mean(dim=-2)
                codebook_ids[block_idx] = -1

            # update metrics
            # self.counts.update(codebook_ids.cpu().numpy().flat)
            if self.logging:
                with torch.no_grad():
                    #                    self.codes_triggered, counts = torch.unique(
                    #                        codebook_ids.cpu(), sorted=False, return_counts=True
                    #                    )
                    #                    self.counts[self.codes_triggered] += counts
                    self.codes_triggered = torch.unique(
                        codebook_ids.cpu(), sorted=False, return_counts=False
                    )
                    self.counts[self.codes_triggered] += 1
                coeff = x.shape[0] * x.shape[1]
                coeff /= self.tokens_processed + x.shape[0] * x.shape[1]
                mse = torch.mean(((x - output) ** 2).sum(dim=-1))
                self.reconstruction_mse += coeff * (
                    mse.item() - self.reconstruction_mse
                )
                self.input_norm += coeff * (
                    torch.norm(x, dim=-1).mean().item() - self.input_norm
                )
                self.output_norm += coeff * (
                    torch.norm(output, dim=-1).mean().item() - self.output_norm
                )
                self.tokens_processed += x.shape[0] * x.shape[1]

            if self.hook_fn is not None:
                self.hook_fn(self.key, codebook_ids.cpu().numpy())
        else:
            # NOTE: was previously doing a gumbel softmax,
            # but found this was not necessary
            # codebook_weights = torch.nn.functional.gumbel_softmax(
            #   logits, hard=False, tau=tau)
            logits = torch.matmul(x, self.codebook.weight.T)
            codebook_weights = torch.nn.functional.softmax(logits, dim=-1)

            # Perform a soft average over the codebook vectors.
            # [batch_size, codebook_size, 1] * [1, codebook_size, dim]
            output = codebook_weights.unsqueeze(-1) * self.codebook.weight.unsqueeze(0)

            output = output.sum(-2)  # codebook size
        return output

    def get_triggered_codes(self):
        """Return the triggered codes."""
        return self.codebook(self.codes_triggered)

    def set_hook_kwargs(self, **kwargs):
        self.hook_kwargs = {**self.hook_kwargs, **kwargs}
        assert all(
            k in ["disable_topk", "disable_codes", "disable_for_tkns", "keep_k_codes", "disable_sim_idx", "cosine"]
            for k in self.hook_kwargs
        )

    def reset_hook_kwargs(self):
        self.hook_kwargs = {
            "disable_topk": 0,
            "disable_codes": [],
            "disable_for_tkns": "all",
            "keep_k_codes": True,
            "disable_sim_idx": [],
            "cosine": False,
        }

    def reset_metrics(self):
        """Reset the counts of the codebook features."""
        self.counts.zero_()
        self.reconstruction_mse = 0
        self.tokens_processed = 0
        self.input_norm = 0
        self.output_norm = 0

    def avg_norm(self):
        """Return the average norm of the codebook features."""
        return self.codebook.weight.norm(p=2, dim=1).mean().item()

    def max_norm(self):
        """Return the maximum norm of the codebook features."""
        return self.codebook.weight.norm(p=2, dim=1).max().item()

    # TODO: Consider using a fraction for the threshold instead of an absolute number
    def expire_codes(self, threshold: int = 1):
        """re-initialize the codebook features with activation count below threshold.

        Args:
        ----
            threshold: minimum count for feature vector to not get replaced. Defaults to 1.
        """
        underused_codes = set()
        for i in range(self.codebook.weight.size(0)):
            if self.counts[i] < threshold:
                underused_codes.add(i)
        with torch.no_grad():
            weights = torch.rand((len(underused_codes), self.codebook.weight.size(0)))
            weights = weights / weights.sum(1, keepdim=True)
            weights = weights.to(self.codebook.weight.device)
            new_codes = torch.einsum("uc,cd->ud", weights, self.codebook.weight)
            underused_codes = torch.tensor(list(underused_codes)).to(
                self.codebook.weight.device,
            )
            try:
                self.codebook.weight[underused_codes] = new_codes
            except IndexError:
                pass

    def most_common_counts(self):
        """Return the most common codebook feature counts."""

        return self.counts.sort()[0].cpu().numpy()

    def load_data(self, data: torch.Tensor):
        """Load the data for kmeans."""
        assert isinstance(self.codebook, KmeansEmbedding)
        self.codebook.load_data(data)

    def clear_data(self):
        """Clear the data for kmeans."""
        assert isinstance(self.codebook, KmeansEmbedding)
        self.codebook.clear_data()

    def initialize(self):
        """Initialize the codebook with kmeans."""
        assert isinstance(self.codebook, KmeansEmbedding)
        self.codebook.initialize()

    def partial_fit_codebook(self):
        """Update the codebook with the data."""
        assert isinstance(self.codebook, KmeansEmbedding)
        self.codebook.partial_fit()


class CompositionalCodebookLayer2(nn.Module):
    """Module that applies distinct codebooks to chunks of input vectors."""

    def __init__(
        self,
        dim: int,
        num_codes: int,
        key: str,
        kmeans_init=False,
        soft_snap: bool = False,
        snap_fn: BaseSnapFunction = CompostionalEuclideanSnapFunction,
        num_codebooks: int = 1,
        hook_fn: Callable = None,
        device=None,
        dtype=None,
    ):
        """Create the compositional codebook layer.

        Args:
        ----
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            kmeans_init: whether to initialize the codebook with k-means of the data. Defaults to False.
            soft_snap: whether to snap the input using softmax. Defaults to False.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
        """
        super().__init__()
        if dim % num_codebooks != 0:
            raise ValueError(
                "dim must be divisible by num_codebooks. Got dim: {}, num_codebooks: {}".format(
                    dim, num_codebooks
                )
            )
        self.num_codebooks = num_codebooks
        factory_kwargs = {"device": device, "dtype": dtype}
        self.codebook = nn.Parameter(
            torch.empty(
                (num_codebooks, num_codes, dim // num_codebooks), **factory_kwargs
            )
        )
        self.snap_fn = snap_fn
        self.counts = [Counter() for _ in range(num_codebooks)]
        self.key = key
        self.hook_fn = hook_fn

    def reset_parameters(self) -> None:
        """Reset the parameters of the codebooks."""
        nn.init.normal_(self.codebook)

    @property
    def active_codes(self):
        """Return the number of active codes in all the codebooks."""
        return sum(len(counter) for counter in self.counts)

    @property
    def num_codes(self):
        """Return the total number of codes in all the codebooks."""
        return self.num_codebooks * self.codebook.size(1)

    def forward(self, x):
        """Snaps activations to elements in the codebook.

        Args:
        ----
            x: input tensor of shape: (batch_size, seq_len, dim).

        Returns: output with the feature vectors replaced using the compositional codebook.
        """
        assert len(x.shape) == 3
        output, codebook_ids = self.snap_fn.apply(x, self.codebook)
        codebook_ids = codebook_ids.cpu().numpy()
        for i, counter in enumerate(self.counts):
            counter.update(codebook_ids[:, :, i, :].flat)
        if self.hook_fn is not None:
            self.hook_fn(self.key, codebook_ids)
        return output

    def reset_counts(self):
        """Reset the counts of the codebook features."""
        for counter in self.counts:
            counter.clear()

    def avg_norm(self):
        """Return the average norm of the codebook features."""
        return self.codebook.norm(p=2, dim=2).mean().item()

    def max_norm(self):
        """Return the average norm of the codebook features."""
        return self.codebook.norm(p=2, dim=2).max().item()

    def most_common_counts(self):
        """Return the counts of the codebook features."""
        raise NotImplementedError("TODO: Implement this")
        counts = np.zeros((self.num_codes // self.num_codebooks, 1))
        for i, counter in enumerate(self.counts):
            counts[list(self.counts.keys())] = np.array(
                list(self.counts.values())
            ).reshape(-1, 1)
        return counts


class CompositionalCodebookLayer(nn.Module):
    """Module that applies distinct codebooks to chunks of input vectors."""

    def __init__(
        self,
        dim: int,
        num_codes: int,
        key: str,
        kmeans_init=False,
        soft_snap: bool = False,
        snap_fn: BaseSnapFunction = EuclideanSnapFunction,
        num_codebooks: int = 1,
        hook_fn: Callable = None,
        kmeans_kwargs: dict = {},
        kcodes: int = 1,
    ):
        """Create the compositional codebook layer.

        Args:
        ----
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            kmeans_init: whether to initialize the codebook with k-means of the data. Defaults to False.
            soft_snap: whether to snap the input using softmax. Defaults to False.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
        """
        super().__init__()
        if dim % num_codebooks != 0:
            raise ValueError(
                "dim must be divisible by num_codebooks. Got dim: {}, num_codebooks: {}".format(
                    dim, num_codebooks
                )
            )
        self.num_codebooks = num_codebooks
        seed = kmeans_kwargs.get("random_state", 0)
        self.codebook = nn.ModuleList(
            [
                CodebookLayer(
                    dim=dim // num_codebooks,
                    num_codes=num_codes,
                    key=key + f"_ccb{i}",
                    kmeans_init=kmeans_init,
                    soft_snap=soft_snap,
                    snap_fn=snap_fn,
                    hook_fn=hook_fn,
                    kmeans_kwargs={**kmeans_kwargs, "random_state": seed + i},
                    kcodes=kcodes,
                )
                for i in range(num_codebooks)
            ]
        )
        self.hook_fn = hook_fn

    def get_triggered_codes(self):
        """Return the triggered codes of the codebooks."""
        triggered_codes = [codebook.get_triggered_codes() for codebook in self.codebook]
        return torch.cat(triggered_codes, dim=0)

    def enable_logging(self):
        for codebook in self.codebook:
            codebook.enable_logging()

    def disable_logging(self):
        for codebook in self.codebook:
            codebook.disable_logging()

    @property
    def active_codes(self):
        """Return the number of active codes in all the codebooks."""
        return sum(codebook.active_codes for codebook in self.codebook)

    @property
    def num_codes(self):
        """Return the total number of codes in all the codebooks."""
        return sum(codebook.num_codes for codebook in self.codebook)

    @property
    def reconstruction_mse(self):
        """Return the reconstruction mse of the codebooks."""
        return sum(codebook.reconstruction_mse for codebook in self.codebook) / len(
            self.codebook
        )

    @property
    def input_norm(self):
        """Return the input norm of the codebooks."""
        return sum(codebook.input_norm for codebook in self.codebook) / len(
            self.codebook
        )

    @property
    def output_norm(self):
        """Return the output norm of the codebooks."""
        return sum(codebook.output_norm for codebook in self.codebook) / len(
            self.codebook
        )

    def get_most_used_code(self):
        """Return the most used code. Uses the first codebook by default."""
        return self.codebook[0].get_most_used_code()

    def set_hook_fn(self, hook_fn):
        """Set the hook function."""
        self.hook_fn = hook_fn
        for codebook in self.codebook:
            codebook.set_hook_fn(hook_fn)

    def forward(self, x):
        """Snaps activations to elements in the codebook.

        Args:
        ----
            x: input tensor of shape: (batch_size, seq_len, dim) or (batch_size, seq_len, num_heads, dim).

        Returns: output with the feature vectors replaced using the compositional codebook.
        """
        if len(x.shape) == 4:
            assert x.shape[2] == self.num_codebooks, f"{x.shape}; num_cbs: {self.num_codebooks}"
            output = torch.stack(
                [self.codebook[i](x[:, :, i]) for i in range(self.num_codebooks)], dim=2
            )
            return output
        else:
            assert len(x.shape) == 3
            output = torch.cat(
                [
                    self.codebook[i](chunk)
                    for i, chunk in enumerate(x.chunk(self.num_codebooks, dim=-1))
                ],
                dim=-1,
            )
            return output

    def set_hook_kwargs(self, head_idx=None, **kwargs):
        if head_idx is not None:
            if type(head_idx) == int:
                head_idx = [head_idx]
            for i in head_idx:
                self.codebook[i].set_hook_kwargs(**kwargs)
            return
        for codebook in self.codebook:
            codebook.set_hook_kwargs(**kwargs)

    def reset_hook_kwargs(self):
        for codebook in self.codebook:
            codebook.reset_hook_kwargs()

    def reset_metrics(self):
        """Reset the metrics stored in the codebooks."""
        for codebook in self.codebook:
            codebook.reset_metrics()

    def avg_norm(self):
        """Return the average norm of the codebook features."""
        return (
            sum(codebook.avg_norm() for codebook in self.codebook) / self.num_codebooks
        )

    def max_norm(self):
        """Return the average norm of the codebook features."""
        return max(codebook.max_norm() for codebook in self.codebook)

    def most_common_counts(self):
        """Return the counts of the codebook features."""
        # num_codes contains total codes across compositional codebooks
        counts = np.zeros(self.num_codes // self.num_codebooks)
        for codebook in self.codebook:
            counts += codebook.most_common_counts()
        return counts

    def load_data(self, data: torch.Tensor):
        """Load data into the codebook."""
        for i, chunk in enumerate(data.chunk(self.num_codebooks, dim=-1)):
            self.codebook[i].load_data(chunk)

    def clear_data(self):
        """Clear the data stored in the codebook."""
        for codebook in self.codebook:
            codebook.clear_data()

    def initialize(self):
        """Initialize the codebook using kmeans."""
        for codebook in self.codebook:
            codebook.initialize()

    def partial_fit_codebook(self):
        """Partially fit the codebook using kmeans."""
        for codebook in self.codebook:
            codebook.partial_fit_codebook()


class GroupedCodebookLayer(nn.Module):
    """Module that applies distinct codebooks to chunks of input vectors."""

    def __init__(
        self,
        dim: int,
        num_codes: int,
        key: str,
        kmeans_init=False,
        soft_snap: bool = False,
        snap_fn: BaseSnapFunction = EuclideanSnapFunction,
        num_codebooks: int = 1,
        hook_fn: Callable = None,
        kmeans_kwargs: Dict = {},
    ):
        super().__init__()
        if num_codes % num_codebooks != 0:
            raise ValueError(
                "num_codes must be divisible by num_codebooks."
                f" Got num_codes: {num_codes}, num_codebooks: {num_codebooks}"
            )
        seed = kmeans_kwargs.get("random_state", 0)
        self.codebook = nn.ModuleList(
            [
                CodebookLayer(
                    dim=dim,
                    num_codes=num_codes // num_codebooks,
                    key=key + f"_gcb{i}",
                    kmeans_init=kmeans_init,
                    soft_snap=soft_snap,
                    snap_fn=snap_fn,
                    hook_fn=hook_fn,
                    kmeans_kwargs={**kmeans_kwargs, "random_state": seed + i},
                )
                for i in range(num_codebooks)
            ]
        )
        self.num_codebooks = num_codebooks
        self.hook_fn = hook_fn
        self.key = key

    def get_triggered_codes(self):
        """Return the triggered codes of the codebooks."""
        triggered_codes = [codebook.get_triggered_codes() for codebook in self.codebook]
        return torch.cat(triggered_codes, dim=0)

    @property
    def active_codes(self):
        """Return the number of active codes in all the codebooks."""
        return sum(codebook.active_codes for codebook in self.codebook)

    @property
    def num_codes(self):
        """Return the total number of codes in all the codebooks."""
        return sum(codebook.num_codes for codebook in self.codebook)

    @property
    def reconstruction_mse(self):
        """Return the reconstruction mse of the codebooks."""
        return sum(codebook.reconstruction_mse for codebook in self.codebook) / len(
            self.codebook
        )

    @property
    def input_norm(self):
        """Return the input norm of the codebooks."""
        return sum(codebook.input_norm for codebook in self.codebook) / len(
            self.codebook
        )

    @property
    def output_norm(self):
        """Return the output norm of the codebooks."""
        return sum(codebook.output_norm for codebook in self.codebook) / len(
            self.codebook
        )

    def get_most_used_code(self):
        """Return the most used code. Uses the first codebook by default."""
        return self.codebook[0].get_most_used_code()

    def set_hook_fn(self, hook_fn):
        """Set the hook function."""
        self.hook_fn = hook_fn
        for codebook in self.codebook:
            codebook.set_hook_fn(hook_fn)

    def forward(self, x):
        """Snaps activations to elements in the codebook.

        Args:
        ----
            x: input tensor of shape: (batch_size, seq_len, dim).

        Returns: output with the feature vectors replaced using the compositional codebook.
        """
        assert len(x.shape) == 3
        output = torch.stack(
            [codebook(x) for codebook in self.codebook],
            dim=0,
        )
        output = output.mean(dim=0)
        return output

    def reset_metrics(self):
        """Reset the metrics stored in the codebooks."""
        for codebook in self.codebook:
            codebook.reset_metrics()

    def avg_norm(self):
        """Return the average norm of the codebook features."""
        return (
            sum(codebook.avg_norm() for codebook in self.codebook) / self.num_codebooks
        )

    def max_norm(self):
        """Return the average norm of the codebook features."""
        return max(codebook.max_norm() for codebook in self.codebook)

    def most_common_counts(self):
        """Return the counts of the codebook features."""
        # num_codes contains total codes across compositional codebooks
        counts = np.zeros(self.num_codes // self.num_codebooks)
        for codebook in self.codebook:
            counts += codebook.most_common_counts()
        return counts

    def load_data(self, data: torch.Tensor):
        """Load data into the codebook."""
        for codebook in self.codebook:
            codebook.load_data(data)

    def clear_data(self):
        """Clear the data stored in the codebook."""
        for codebook in self.codebook:
            codebook.clear_data()

    def initialize(self):
        """Initialize the codebook using kmeans."""
        for codebook in self.codebook:
            codebook.initialize()

    def partial_fit_codebook(self):
        """Partially fit the codebook using kmeans."""
        for codebook in self.codebook:
            codebook.partial_fit_codebook()


class CodebookWrapper(nn.Module, abc.ABC):
    """Wraps a nn module by applying codebooks on the output of the layer."""

    def __init__(
        self,
        module_layer: nn.Module,
        codebook_cls: Union[
            Type[CodebookLayer],
            Type[CompositionalCodebookLayer],
            Type[GroupedCodebookLayer],
        ],
        dim: int,
        num_codes: int,
        key: str,
        snap_fn: BaseSnapFunction = EuclideanSnapFunction,
        num_codebooks: int = 1,
        hook_fn: Callable = None,
        **kwargs,
    ):
        """Create the transformer layer wrapped with the codebook.

        Args:
        ----
            transformer_layer (_type_): _description_
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
        """
        super().__init__()
        self.module_layer = module_layer
        kwargs.update(key=key, snap_fn=snap_fn, hook_fn=hook_fn)
        if codebook_cls != CodebookLayer:
            kwargs["num_codebooks"] = num_codebooks
        self.codebook_layer = codebook_cls(
            dim,
            num_codes,
            **kwargs,
        )
        self.snap = True
        self._store_data = False

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def store_data(self):
        """Context manager to initialize codebooks using kmeans."""
        self._store_data = True

    def initialize_codebooks(self):
        """Initialize the codebooks using kmeans."""
        self.codebook_layer.initialize()
        self._store_data = False

    def __getattr__(self, name):
        """Get attribute from the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module_layer, name)


class TransformerLayerWrapper(CodebookWrapper):
    """Wraps a transformer layer module by applying codebooks on the output of the layer."""

    def __init__(
        self,
        module_layer: nn.Module,
        codebook_cls: Union[
            Type[CodebookLayer],
            Type[CompositionalCodebookLayer],
            Type[GroupedCodebookLayer],
        ],
        dim: int,
        num_codes: int,
        key: str,
        snap_fn: BaseSnapFunction = EuclideanSnapFunction,
        num_codebooks: int = 1,
        hook_fn: Callable = None,
        **kwargs,
    ):
        """Create the transformer layer wrapped with the codebook.

        Args:
        ----
            module_layer (_type_): _description_
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
        """
        super().__init__(
            module_layer=module_layer,
            codebook_cls=codebook_cls,
            dim=dim,
            num_codes=num_codes,
            key=key,
            snap_fn=snap_fn,
            num_codebooks=num_codebooks,
            hook_fn=hook_fn,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        """Forward function for the wrapped layer.

        Returns: output using codebook features if `snap` is enabled otherwise
            returns the output of the transformer layer.
        """
        layer_outputs = self.module_layer(*args, **kwargs)
        tensor_output = layer_outputs
        if isinstance(layer_outputs, tuple):
            tensor_output = layer_outputs[0]
        if self._store_data:
            self.codebook_layer.load_data(tensor_output)
        if self.snap:
            tensor_output = self.codebook_layer(tensor_output)
            if isinstance(layer_outputs, tuple):
                layer_outputs = (tensor_output, *layer_outputs[1:])
            else:
                layer_outputs = tensor_output

        return layer_outputs


class MLPWrapper(CodebookWrapper):
    """Wraps a MLP layer module by applying codebooks on the output of the layer."""

    def __init__(
        self,
        module_layer: nn.Module,
        codebook_cls: Union[
            Type[CodebookLayer],
            Type[CompositionalCodebookLayer],
            Type[GroupedCodebookLayer],
        ],
        dim: int,
        num_codes: int,
        key: str,
        snap_fn: BaseSnapFunction = EuclideanSnapFunction,
        num_codebooks: int = 1,
        hook_fn: Callable = None,
        **kwargs,
    ):
        """Create the transformer layer wrapped with the codebook.

        Args:
        ----
            module_layer:
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
        """
        super().__init__(
            module_layer=module_layer,
            codebook_cls=codebook_cls,
            dim=dim,
            num_codes=num_codes,
            key=key,
            snap_fn=snap_fn,
            num_codebooks=num_codebooks,
            hook_fn=hook_fn,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        """Forward function for the wrapped layer.

        Returns: output using codebook features if `snap` is enabled otherwise
            returns the output of the transformer layer.
        """
        layer_outputs = self.module_layer(*args, **kwargs)
        if self._store_data:
            self.codebook_layer.load_data(layer_outputs)
        if self.snap:
            layer_outputs = self.codebook_layer(layer_outputs)

        return layer_outputs


class CodebookModelConfig(transformers.PretrainedConfig):
    model_type = "codebook"

    def __init__(
        self,
        codebook_type: Union[str, Sequence] ="vanilla",
        num_codes: int = 100,
        num_codebooks: Union[int, Sequence] = 1,
        layers_to_snap: Sequence = (),
        similarity_metric: str = "inner_product",
        codebook_at: Union[str, Sequence] = "mlp",
        loss: str = "base",
        k_codebook: Union[int, Sequence] = 1,
        kmeans_init: bool = False,
        kmeans_init_examples: int = 1000,
        kmeans_path: str = None,
        kmeans_kwargs: dict = {},
        **kwargs,
    ) -> None:
        """Create the config for the codebook model.

        Args:
        ----
            num_codes: number of codebook features to have.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
            layers_to_snap: Index of transformer layers in the model on which codebook to apply.
                Defaults to []. Can contain negative numbers to index from the last layers.
            similarity_metric: similarity metric to use. Can be either 'euclidean' (default) or 'inner_product'.
            codebook_at: where to apply codebook. Can be either 'mlp' (default) or 'transformer_block'.
            loss: whether to use the loss used in VQVAE paper or the CLM loss.
            k_codebook: number of nearest neighbors in codebook snapping.
        """
        super().__init__(**kwargs)
        if type(codebook_type) == str:
            codebook_type = [codebook_type]
        if type(num_codebooks) == int:
            num_codebooks = [num_codebooks] * len(codebook_type)
        if type(k_codebook) == int:
            k_codebook = [k_codebook] * len(codebook_type)
        for i in range(len(codebook_type)):
            if codebook_type[i] not in ["vanilla", "compositional", "grouped"]:
                raise ValueError(f"Invalid codebook type {i_cb_type}")
            if codebook_type[i] == "vanilla" and num_codebooks[i] != 1:
                raise ValueError("Vanilla codebook type can only have 1 codebook.")

        if loss.split("-")[0] not in ["base", "aeloss", "fullaeloss", "vqvae"]:
            raise ValueError(f"Invalid loss {loss}")
        if similarity_metric not in ["euclidean", "inner_product"]:
            raise ValueError(f"Invalid similarity metric {similarity_metric}")

        self.codebook_type = codebook_type
        self.num_codes = num_codes
        self.num_codebooks = num_codebooks
        self.layers_to_snap = layers_to_snap
        self.similarity_metric = similarity_metric
        self.codebook_at = codebook_at
        if isinstance(codebook_at, str):
            self.codebook_at = codebook_at.lower().split(",")
        self.loss = loss
        self.k_codebook = k_codebook
        self.kmeans_init = kmeans_init
        self.kmeans_path = kmeans_path
        self.kmeans_init_examples = kmeans_init_examples
        self.kmeans_kwargs = kmeans_kwargs


class CodebookModel(transformers.PreTrainedModel, abc.ABC):
    """ABC for a model containing codebook features."""

    config_class = CodebookModelConfig

    def __init__(
        self,
        config: CodebookModelConfig,
        model: nn.Module,
    ) -> None:
        """Build the codebook based model.

        Args:
        ----
            config: config for the model.
            model: torch model to apply codebooks to.
        """
        super().__init__(config=config)
        self.codebook_cls = []
        for cb_type in config.codebook_type:
            if cb_type == "vanilla":
                self.codebook_cls.append(CodebookLayer)
            elif cb_type == "compositional":
                self.codebook_cls.append(CompositionalCodebookLayer)
            elif cb_type == "grouped":
                self.codebook_cls.append(GroupedCodebookLayer)

        self.model = model
        self.model_params = list(model.parameters())
        for i in range(len(self.config.num_codebooks)):
            if self.config.num_codebooks[i] == -1:
                self.config.num_codebooks[i] = self.num_heads
        num_layers = self.num_layers()
        if (
            type(self.config.layers_to_snap) is str
            and self.config.layers_to_snap == "all"
        ):
            self.config.layers_to_snap = list(range(num_layers))
        else:
            self.config.layers_to_snap = list(self.config.layers_to_snap)
            for i in range(len(self.config.layers_to_snap)):
                assert (
                    -num_layers <= i and i < num_layers
                ), f"Invalid layer index {i}. Layer index should be between {-num_layers} and {num_layers - 1}."
                if self.config.layers_to_snap[i] < 0:
                    self.config.layers_to_snap[i] += num_layers
        self.config.layers_to_snap = sorted(self.config.layers_to_snap)
        self.codebook_params = []
        self.all_codebook_wrappers = {}
        # self.freeze_model_params()
        BaseSnapFunction.loss = self.config.loss
        if self.config.similarity_metric == "euclidean":
            self.snap_fn = EuclideanSnapFunction
        elif self.config.similarity_metric == "inner_product":
            self.snap_fn = InnerProductSnapFunction
        else:
            raise ValueError(
                "`similarity_metric` should be either 'euclidean' or 'inner_product'."
            )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
        
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @property
    def device(self):
        return self.model.device

    @property
    def all_codebooks(self):
        return {
            k: [v.codebook_layer for v in vl]
            for k, vl in self.all_codebook_wrappers.items()
        }

    def add_codebooks(self):
        """Adds codebooks for the layers that are to be snapped."""
        layers = self.layers()
        for i in range(len(layers)):
            if i in self.config.layers_to_snap:
                codebooks_in_layer = []
                for i_cb, cb_at in enumerate(self.config.codebook_at):
                    if cb_at == "transformer_block":
                        self.codebook_at_transformer(layers, i, codebooks_in_layer, i_cb)
                    if cb_at == "mlp":
                        self.codebook_at_mlp(layers, i, codebooks_in_layer, i_cb)
                    if cb_at == "mlp_mid":
                        self.codebook_at_mlp_mid(layers, i, codebooks_in_layer, i_cb)
                    if cb_at == "qkv":
                        self.codebook_at_qkv(layers, i, codebooks_in_layer, i_cb)
                    if cb_at == "attention":
                        self.codebook_at_attention(layers, i, codebooks_in_layer, i_cb)
                    if cb_at == "preproj_attention":
                        self.codebook_at_preprojection_attn(layers, i, codebooks_in_layer, i_cb)
                    if cb_at == "attention_and_mlp":
                        self.codebook_at_attention_plus_mlp(layers, i, codebooks_in_layer, i_cb)

                if len(codebooks_in_layer) == 0:
                    raise ValueError(
                        f"Invalid value for `codebook_at`: {self.config.codebook_at}."
                    )
                self.all_codebook_wrappers[i] = codebooks_in_layer

    def codebook_at_transformer(self, layers, i, codebooks_in_layer, i_cb):
        layers[i] = TransformerLayerWrapper(
            layers[i],
            codebook_cls=self.codebook_cls[i_cb],
            dim=self.d_model,
            num_codes=self.config.num_codes,
            key=f"layer{i}_tb",
            snap_fn=self.snap_fn,
            num_codebooks=self.config.num_codebooks[i_cb],
            kmeans_init=self.config.kmeans_init,
            kmeans_kwargs=self.config.kmeans_kwargs,
            kcodes=self.config.k_codebook[i_cb],
        )
        self.codebook_params += list(
            layers[i].codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(layers[i])

    def codebook_at_mlp(self, layers, i, codebooks_in_layer, i_cb):
        wrapped_mlp = MLPWrapper(
            layers[i].__getattr__(self.mlp_key),
            codebook_cls=self.codebook_cls[i_cb],
            dim=self.d_model,
            num_codes=self.config.num_codes,
            key=f"layer{i}_mlp",
            snap_fn=self.snap_fn,
            num_codebooks=self.config.num_codebooks[i_cb],
            kmeans_init=self.config.kmeans_init,
            kmeans_kwargs=self.config.kmeans_kwargs,
            kcodes=self.config.k_codebook[i_cb],
        )
        layers[i].__setattr__(self.mlp_key, wrapped_mlp)
        self.codebook_params += list(
            wrapped_mlp.codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(wrapped_mlp)

    def codebook_at_mlp_mid(self, layers, i, codebooks_in_layer, i_cb):
        mlp = layers[i].__getattr__(self.mlp_key)
        wrapped_hidden_layer = MLPWrapper(
            mlp.__getattr__(self.mlp_mid_key),
            codebook_cls=self.codebook_cls[i_cb],
            dim=self.itermediate_size(),
            num_codes=self.config.num_codes,
            key=f"layer{i}_mlp_mid",
            snap_fn=self.snap_fn,
            num_codebooks=self.config.num_codebooks[i_cb],
            kmeans_init=self.config.kmeans_init,
            kmeans_kwargs=self.config.kmeans_kwargs,
            kcodes=self.config.k_codebook[i_cb],
        )
        mlp.__setattr__(self.mlp_mid_key, wrapped_hidden_layer)
        self.codebook_params += list(
            wrapped_hidden_layer.codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(wrapped_hidden_layer)

    def codebook_at_qkv(self, layers, i, codebooks_in_layer, i_cb):
        attn = layers[i].__getattr__(self.attention_key)
        qkv = attn.__getattr__(self.qkv_key)
        wrapped_hidden_layer = MLPWrapper(
            qkv,
            codebook_cls=CompositionalCodebookLayer,
            dim=3 * self.d_model,
            num_codes=self.config.num_codes,
            key=f"layer{i}_qkv",
            snap_fn=self.snap_fn,
            num_codebooks=3 * self.config.num_codebooks[i_cb],
            kmeans_init=self.config.kmeans_init,
            kmeans_kwargs=self.config.kmeans_kwargs,
            kcodes=self.config.k_codebook[i_cb],
        )
        attn.__setattr__(self.qkv_key, wrapped_hidden_layer)
        self.codebook_params += list(
            wrapped_hidden_layer.codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(wrapped_hidden_layer)

    def codebook_at_attention(self, layers, i, codebooks_in_layer, i_cb):
        wrapped_attn = TransformerLayerWrapper(
            layers[i].__getattr__(self.attention_key),
            codebook_cls=self.codebook_cls[i_cb],
            dim=self.d_model,
            num_codes=self.config.num_codes,
            key=f"layer{i}_attn",
            snap_fn=self.snap_fn,
            num_codebooks=self.config.num_codebooks[i_cb],
            kmeans_init=self.config.kmeans_init,
            kmeans_kwargs=self.config.kmeans_kwargs,
            kcodes=self.config.k_codebook[i_cb],
        )
        layers[i].__setattr__(self.attention_key, wrapped_attn)
        self.codebook_params += list(
            wrapped_attn.codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(wrapped_attn)

    def codebook_at_preprojection_attn(self, layers, i, codebooks_in_layer, i_cb):
        codebook = self.codebook_cls[i_cb](
            dim=self.d_model,
            num_codes=self.config.num_codes,
            key=f"layer{i}_attn_preproj",
            snap_fn=self.snap_fn,
            num_codebooks=self.config.num_codebooks[i_cb],
            kmeans_init=self.config.kmeans_init,
            kmeans_kwargs=self.config.kmeans_kwargs,
            kcodes=self.config.k_codebook[i_cb],
        )
        new_block = self.pre_projection_attn_codebook_cls(
            self.base_model_cfg(),
            i,
            codebook,
        )
        codebooks_in_layer.append(new_block)
        self.codebook_params += list(codebook.parameters())
        new_block.load_state_dict(
            layers[i].__getattr__(self.attention_key).state_dict(), strict=False
        )
        layers[i].__setattr__(self.attention_key, new_block)

    def codebook_at_attention_plus_mlp(self, layers, i, codebooks_in_layer, i_cb):
        codebook = self.codebook_cls[i_cb](
            dim=self.d_model,
            num_codes=self.config.num_codes,
            key=f"layer{i}_attn+mlp",
            snap_fn=self.snap_fn,
            num_codebooks=self.config.num_codebooks[i_cb],
            kmeans_init=self.config.kmeans_init,
            kmeans_kwargs=self.config.kmeans_kwargs,
            kcodes=self.config.k_codebook[i_cb],
        )
        pre_res_block = self.pre_residual_codebook_cls(
            self.base_model_cfg(),
            i,
            codebook,
        )
        codebooks_in_layer.append(pre_res_block)
        self.codebook_params += list(codebook.parameters())
        pre_res_block.load_state_dict(layers[i].state_dict(), strict=False)
        layers[i] = pre_res_block

    def reset_codebook_metrics(self):
        """Resets the metrics stored of the codebooks."""
        for i, layers in self.all_codebooks.items():
            assert i in self.config.layers_to_snap
            for layer in layers:
                layer.reset_metrics()

    def enable_codebooks(self):
        """Enable the use of codebooks in all the layers to snap."""
        for i, layers in self.all_codebook_wrappers.items():
            assert i in self.config.layers_to_snap
            for layer in layers:
                layer.snap = True

    def disable_codebooks(self):
        """Disable the use of codebooks in all the layers."""
        for i, layers in self.all_codebook_wrappers.items():
            assert i in self.config.layers_to_snap
            for layer in layers:
                layer.snap = False

    def get_codebook_params(self):
        """Gets codebook parameters."""
        return self.codebook_params

    def get_model_params(self):
        """Gets model's original parameters (not including codebook params)."""
        return self.model_params

    def set_hook_kwargs(self, idx=None, **kwargs):
        if idx is not None:
            if type(idx) == int:
                idx = [idx]
            for i in idx:
                layers = self.all_codebooks[i]
                for layer in layers:
                    layer.set_hook_kwargs(**kwargs)
            return
        for i, layers in self.all_codebooks.items():
            for layer in layers:
                layer.set_hook_kwargs(**kwargs)

    def reset_hook_kwargs(self, idx=None):
        if idx is not None:
            for layer in list(self.all_codebooks.values())[idx]:
                layer.reset_hook_kwargs()
            return
        for i, layers in self.all_codebooks.items():
            for layer in layers:
                layer.reset_hook_kwargs()

    def set_hook_fn(self, hook_fn: Callable):
        """Sets the hook function to be called after every forward pass of every codebook layer."""
        for i, layers in self.all_codebooks.items():
            assert i in self.config.layers_to_snap
            for layer in layers:
                layer.set_hook_fn(hook_fn)

    def get_triggered_codes(self):
        """Gets the codes triggered in the last forward pass."""
        triggered_codes = []
        for i, layers in self.all_codebooks.items():
            for layer in layers:
                triggered_codes.append(layer.get_triggered_codes())
        triggered_codes = torch.cat(triggered_codes, dim=0)
        assert triggered_codes.shape[1] * self.config.num_codebooks == self.d_model
        return triggered_codes

    def codebook_regularization(self, p=1):
        """Regularizer for codebook weights."""
        triggered_codes = self.get_triggered_codes()
        reg = triggered_codes.norm(p=p, dim=1).sum()
        return reg

    def get_input_embeddings(self):
        """Gets input embeddings of the model."""
        return self.model.get_input_embeddings()

    def save_kmeans_embeddings(self, path):
        """Saves kmeans embeddings to a file."""
        state_dict = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if "codebook_layer" in k}
        torch.save(state_dict, path)

    def load_kmeans_embeddings(self, path):
        """Loads kmeans embeddings from a file."""
        state_dict = torch.load(path)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if "codebook_layer" in k]
        assert len(unexpected) == 0 and len(missing) == 0

    def init_codebook(self, dataloader):
        """Initializes the codebook weights using kmeans."""
        # check if `kmeans_path` exists and load kmeans embeddings
        if self.config.kmeans_path and os.path.exists(self.config.kmeans_path):
            self.load_kmeans_embeddings(self.config.kmeans_path)
            return

        print("Running kmeans initialization for all the codebooks...")

        self.model.eval()
        self.to(torch.device("cuda"))
        self.disable_codebooks()

        # enable loading training data for kmeans initialization
        for codebooks in self.all_codebook_wrappers.values():
            for codebook in codebooks:
                codebook.store_data()

        # load data and fit kmeans model
        examples = 0
        for data in tqdm(dataloader):
            if examples >= self.config.kmeans_init_examples:
                break
            examples += data["input_ids"].shape[0]
            data = {k: v.to(self.device) for k, v in data.items()}
            self.model(**data)
            self.partial_fit_codebook()

        # disable loading data and initialize codebook weights
        for codebooks in self.all_codebook_wrappers.values():
            for codebook in codebooks:
                codebook.initialize_codebooks()

        if self.config.kmeans_path and not os.path.exists(self.config.kmeans_path):
            self.save_kmeans_embeddings(self.config.kmeans_path)

        self.enable_codebooks()

    def partial_fit_codebook(self):
        """Fits the codebook to the data stored in the codebook layer."""
        for codebooks in self.all_codebooks.values():
            for codebook in codebooks:
                codebook.partial_fit_codebook()

    def enable_logging(self):
        for codebooks in self.all_codebooks.values():
            for codebook in codebooks:
                codebook.enable_logging()

    def disable_logging(self):
        for codebooks in self.all_codebooks.values():
            for codebook in codebooks:
                codebook.disable_logging()

    @abc.abstractmethod
    def itermediate_size(self):
        """Returns the intermediate size of the model."""
        pass

    @property
    @abc.abstractmethod
    def mlp_mid_key(self):
        """Returns the key of layer in MLP layer where codebook is to be applied."""
        pass

    @property
    @abc.abstractmethod
    def d_model(self):
        """Returns the dimension of the model."""
        pass

    @abc.abstractmethod
    def layers(self):
        """Returns the list of transformer layers of the model."""
        pass

    @abc.abstractmethod
    def num_layers(self):
        """Returns the number of transformer layers in the model."""
        pass

    @abc.abstractmethod
    def base_model_cfg(self):
        """Returns the base model config."""
        pass


class BertCodebookModel(CodebookModel):
    """Codebook model for Bert-based models."""

    def __init__(
        self,
        config,
        model,
    ):
        """Build the codebook based model.

        Args:
        ----
            config: config for the model.
            model: bert model to apply codebooks to.
        """
        super().__init__(
            config=config,
            model=model,
        )
        self.add_codebooks()
        self.forward = self.model.forward

    # def forward(self, *args, **kwargs):
    #     return self.model(*args, **kwargs)

    def layers(self):
        """Returns the list of transformer layers of the model."""
        return self.model.bert.encoder.layer

    def num_layers(self):
        """Returns the number of transformer layers in the model."""
        return self.base_model_cfg().num_hidden_layers


class GPT2CodebookModel(CodebookModel):
    """Codebook model for GPT2."""

    def __init__(
        self,
        config,
        model,
    ):
        """Build the codebook based model.

        Args:
        ----
            config: config for the model.
            model: GPT2 model to apply codebooks to.

        """
        super().__init__(
            config=config,
            model=model,
        )
        self.add_codebooks()
        self.forward = self.model.forward

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "model":
            self.forward = self.model.forward

    def forward(self, *args, labels: Optional[torch.LongTensor] = None, **kwargs):
        raise RuntimeError(
            "This shouldn't get executed as forward is overridden in init."
        )

    def layers(self):
        """Returns the list of transformer layers of the model."""
        return self.model.transformer.h

    def num_layers(self):
        """Returns the number of transformer layers in the model."""
        return self.model.config.n_layer

    def resize_token_embeddings(self, new_num_tokens):
        """Resizes token embeddings of the model."""
        return self.model.resize_token_embeddings(new_num_tokens)

    def itermediate_size(self):
        """Returns the intermediate size of the model."""
        if self.model.config.n_inner is not None:
            return self.model.config.n_inner
        else:
            return 4 * self.d_model

    @property
    def attention_key(self):
        """Returns the attribute name used for attention in the model."""
        return "attn"

    @property
    def qkv_key(self):
        return "c_attn"

    @property
    def mlp_key(self):
        """Returns the attribute name used for mlp in the model."""
        return "mlp"

    @property
    def mlp_mid_key(self):
        """Returns the attribute name used for mlp hidden layer in the model."""
        return "act"

    @property
    def pre_projection_attn_codebook_cls(self):
        """Returns the class to use for applying codebook to attention before projection."""
        return mod_model_classes.PreProjectionAttentionCodebookGPT2

    @property
    def pre_residual_codebook_cls(self):
        """Returns the class to use for codebook before residual."""
        return mod_model_classes.PreResidualCodebookGPT2Block

    @property
    def num_heads(self):
        """Returns the number of heads in the model."""
        return self.model.config.n_head

    @property
    def d_model(self):
        """Returns the dimension of the model."""
        return self.model.config.hidden_size

    def base_model_cfg(self):
        """Returns the base model config."""
        return self.model.config


class GPTNeoXCodebookModel(CodebookModel):
    """Codebook model for GPT2."""

    def __init__(
        self,
        config,
        model,
    ):
        """Build the codebook based model.

        Args:
        ----
            config: config for the model.
            model: GPT2 model to apply codebooks to.
        """
        super().__init__(
            config=config,
            model=model,
        )
        self.add_codebooks()
        self.forward = self.model.forward

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "model":
            self.forward = self.model.forward

    def forward(self, *args, labels: Optional[torch.LongTensor] = None, **kwargs):
        raise RuntimeError(
            "This shouldn't get executed as forward is overridden in init."
        )

    def layers(self):
        """Returns the list of transformer layers of the model."""
        return self.model.gpt_neox.layers

    def num_layers(self):
        """Returns the number of transformer layers in the model."""
        return self.model.config.num_hidden_layers

    def resize_token_embeddings(self, new_num_tokens):
        """Resizes token embeddings of the model."""
        raise NotImplementedError("Not implemented for GPTNeoX.")

    def itermediate_size(self):
        """Returns the intermediate size of the model."""
        return self.model.config.intermediate_size

    @property
    def attention_key(self):
        """Returns the attribute name used for attention in the model."""
        return "attention"

    @property
    def qkv_key(self):
        return "query_key_value"

    @property
    def mlp_key(self):
        """Returns the attribute name used for mlp in the model."""
        return "mlp"

    @property
    def mlp_mid_key(self):
        """Returns the attribute name used for mlp hidden layer in the model."""
        return "act"

    @property
    def pre_projection_attn_codebook_cls(self):
        """Returns the class to use for applying codebook to attention before projection."""
        return mod_model_classes.PreProjectionAttentionCodebookGPTNeoX

    @property
    def pre_residual_codebook_cls(self):
        """Returns the class to use for codebook before residual."""
        return mod_model_classes.PreResidualCodebookGPTNeoXBlock

    @property
    def num_heads(self):
        """Returns the number of heads in the model."""
        return self.model.config.num_attention_heads

    @property
    def d_model(self):
        """Returns the dimension of the model."""
        return self.model.config.hidden_size

    def base_model_cfg(self):
        """Returns the base model config."""
        return self.model.config


class HookedTransformerCodebookModel(CodebookModel):
    """Codebook model for HookedTransformer."""

    def __init__(
        self,
        config,
        model,
        base_model_config,
    ):
        """Build the codebook based model.

        Args:
        ----
            config: config for the model.
            model: GPT2 model to apply codebooks to.
        """
        super().__init__(
            config=config,
            model=model,
        )
        # get config key values from original class
        for k, v in base_model_config.__dict__.items():
            if k not in self.model.cfg.__dict__:
                self.model.cfg.__setattr__(k, v)
        for k1, k2 in base_model_config.attribute_map.items():
            if k1 not in self.model.cfg.__dict__:
                self.model.cfg.__setattr__(k1, base_model_config.__getattribute__(k2))
        self.add_codebooks()
        self.forward = self.model.forward

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "model":
            self.forward = self.model.forward

    def forward(self, *args, labels: Optional[torch.LongTensor] = None, **kwargs):
        raise RuntimeError(
            "This shouldn't get executed as forward is overridden in init."
        )

    def layers(self):
        """Returns the list of transformer layers of the model."""
        return self.model.blocks

    def num_layers(self):
        """Returns the number of transformer layers in the model."""
        return self.model.cfg.n_layers

    def resize_token_embeddings(self, new_num_tokens):
        """Resizes token embeddings of the model."""
        raise NotImplementedError("Not implemented for HookedTransformer.")

    def itermediate_size(self):
        """Returns the intermediate size of the model."""
        return self.model.cfg.d_mlp

    @property
    def attention_key(self):
        """Returns the attribute name used for attention in the model."""
        return "attn"

    @property
    def mlp_key(self):
        """Returns the attribute name used for mlp in the model."""
        return "mlp"

    @property
    def mlp_mid_key(self):
        """Returns the attribute name used for mlp hidden layer in the model."""
        return "act_fn"

    @property
    def pre_projection_attn_codebook_cls(self):
        """Returns the class to use for applying codebook to attention before projection."""
        return mod_model_classes.PreProjectionAttentionCodebookHookedTransformer

    @property
    def pre_residual_codebook_cls(self):
        """Returns the class to use for codebook before residual."""
        if self.model.cfg.original_architecture == "GPT2LMHeadModel":
            return mod_model_classes.PreResidualCodebookGPT2Block
        elif self.model.cfg.original_architecture == "GPTNeoXForCausalLM":
            return mod_model_classes.PreResidualCodebookGPTNeoXBlock
        else:
            raise ValueError(
                f"pre_residual cls not available for {self.model.cfg.model_name}"
            )

    @property
    def num_heads(self):
        """Returns the number of heads in the model."""
        return self.model.cfg.n_heads

    @property
    def d_model(self):
        """Returns the dimension of the model."""
        return self.model.cfg.d_model

    @property
    def device(self):
        return self.model.cfg.device

    def base_model_cfg(self):
        """Returns the base model config."""
        return self.model.cfg


def wrap_codebook(model_or_path, config=None, pretrained_path=None):
    """Wraps a model with codebooks."""
    if isinstance(model_or_path, str):
        model = transformers.AutoModelForCausalLM.from_pretrained(model_or_path)
    elif isinstance(model_or_path, transformers.PreTrainedModel):
        model = model_or_path
    else:
        raise ValueError(
            "`model_or_path` should be either a string or a PreTrainedModel."
        )
    if pretrained_path is not None:
        if model.config.model_type == "gpt2":
            return GPT2CodebookModel.from_pretrained(pretrained_path, model)
        elif model.config.model_type == "gpt_neox":
            return GPTNeoXCodebookModel.from_pretrained(pretrained_path, model)
        elif model.config.model_type == "bert":
            return BertCodebookModel.from_pretrained(pretrained_path, model)
        else:
            raise ValueError(
                f"Model type {model.config.model_type} not supported with codebooks."
            )
    if config is None:
        RuntimeWarning("No config provided. Using default config.")
        config = CodebookModelConfig()
    if model.config.model_type == "gpt2":
        return GPT2CodebookModel(config, model)
    elif model.config.model_type == "gpt_neox":
        return GPTNeoXCodebookModel(config, model)
    elif model.config.model_type == "bert":
        return BertCodebookModel(config, model)
    else:
        raise ValueError(
            f"Model type {model.config.model_type} not supported with codebooks."
        )


def convert_to_hooked_model(model_path, orig_cb_model, hooked_kwargs={}):
    """Wraps a hooked tranformer model with codebooks."""
    model = transformer_lens.HookedTransformer.from_pretrained(
        model_path,
        **hooked_kwargs,
    )
    if "device" in hooked_kwargs:
        hooked_kwargs.pop("device")
    state_dict = convert_state_dict(orig_cb_model.model, model.cfg)
    model.load_and_process_state_dict(
        state_dict,
        **hooked_kwargs,
    )
    cb_model = HookedTransformerCodebookModel(
        orig_cb_model.config, model, orig_cb_model.model.config
    )
    cb_sd = {}

    for key, value in orig_cb_model.model.state_dict().items():
        if "codebook" in key:
            key = key.replace(orig_cb_model.attention_key, cb_model.attention_key)
            key = key.replace(orig_cb_model.mlp_key, cb_model.mlp_key)
            split_key = re.split(r"(\d+)", key)
            split_key[0] = "blocks."
            cb_sd["".join(split_key)] = value
    _, unexpected = cb_model.model.load_state_dict(cb_sd, strict=False)
    assert len(unexpected) == 0
    cb_model.model.setup()
    return cb_model


def convert_to_hooked_model_for_toy(model_path, orig_cb_model, config, hooked_kwargs={}):
    """Wraps a hooked tranformer model with codebooks."""
    import transformer_lens.loading_from_pretrained as loading
    hooked_config = loading.convert_hf_model_config(model_path, config)
    model = transformer_lens.HookedTransformer(hooked_config)
    if "device" in hooked_kwargs:
        hooked_kwargs.pop("device")
    state_dict = convert_state_dict(orig_cb_model.model, model.cfg)
    model.load_and_process_state_dict(
        state_dict,
        **hooked_kwargs,
    )
    cb_model = HookedTransformerCodebookModel(
        orig_cb_model.config, model, orig_cb_model.model.config
    )
    cb_sd = {}

    for key, value in orig_cb_model.model.state_dict().items():
        if "codebook" in key:
            key = key.replace(orig_cb_model.attention_key, cb_model.attention_key)
            key = key.replace(orig_cb_model.mlp_key, cb_model.mlp_key)
            split_key = re.split(r"(\d+)", key)
            split_key[0] = "blocks."
            cb_sd["".join(split_key)] = value
    _, unexpected = cb_model.model.load_state_dict(cb_sd, strict=False)
    assert len(unexpected) == 0
    cb_model.model.setup()
    return cb_model


def convert_state_dict(model, cfg: transformer_lens.HookedTransformerConfig):
    """Converts a state_dict from a HuggingFace model to a state_dict
    compatible with HookedTransformer."""
    if cfg.original_architecture == "GPT2LMHeadModel":
        return transformer_lens.loading.convert_gpt2_weights(model, cfg)
    elif cfg.original_architecture == "GPTNeoForCausalLM":
        return transformer_lens.loading.convert_neo_weights(model, cfg)
    elif cfg.original_architecture == "GPTJForCausalLM":
        return transformer_lens.loading.convert_gptj_weights(model, cfg)
    elif cfg.original_architecture == "GPTNeoXForCausalLM":
        return transformer_lens.loading.convert_neox_weights(model, cfg)
    elif cfg.original_architecture == "OPTForCausalLM":
        return transformer_lens.loading.convert_opt_weights(model, cfg)
    elif cfg.original_architecture == "neel-solu-old":
        return transformer_lens.loading.convert_neel_solu_old_weights(model, cfg)
    elif cfg.original_architecture == "neel":
        return model.state_dict()
    else:
        raise ValueError(f"Unknown architecture {cfg.original_architecture}")
