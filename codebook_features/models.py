"""Model related classes."""

import abc
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import numpy as np
import torch
import transformer_lens
import transformer_lens.loading_from_pretrained as loading
import transformers
from sklearn import cluster as sklearn_cluster
from torch import nn
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint
from vqtorch import nn as vqtorch_nn

from codebook_features import mod_model_classes


class KMeansEmbedding(nn.Embedding):
    """Embedding layer with mini-batch K-Means initialization."""

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
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **kmeans_kwargs,
    ) -> None:
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
            kmeans_kwargs: keyword arguments for sklearn.cluster.MiniBatchKMeans.
        """
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            device=device,
            dtype=dtype,
        )
        self.data: Optional[torch.Tensor] = None
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
        assert self.data is not None, "No data loaded."
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

    This is the base class. It should be subclassed with a forward function.
    """

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
        else:
            raise NotImplementedError(f"Loss {BaseSnapFunction.loss} not implemented.")

        return grad_inputs, grad_codebook, None, None


class InnerProductSnapFunction(BaseSnapFunction):
    """Snap function with inner product as similarity metric."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, codebook: torch.Tensor, kcodes, hook_kwargs):
        """Compute output of the snap function with the maximum inner product as the similarity metric.

        Replaces each dimension vector of input with features from codebook
        having highest dot-product.

        Args:
        ----
            ctx: torch context used for efficiently storing tensors for backward pass.
            inputs: input data.
            codebook: codebook matrix. Shape: (num_features, hidden_dim_size).
            kcodes: number of codebook features to use for computing the output.
            hook_kwargs: dictionary of hook arguments.

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
                codebook_ids = codebook_ids[:, :, -kcodes:]
            else:
                for code in hook_kwargs["disable_codes"]:
                    logits[:, hook_kwargs["disable_for_tkns"], code] = float("-inf")
                _, codebook_ids_all = logits.topk(
                    kcodes + hook_kwargs["disable_topk"], dim=-1
                )
                codebook_ids = codebook_ids_all[:, :, :kcodes]
                codebook_ids[:, hook_kwargs["disable_for_tkns"]] = codebook_ids_all[
                    :, hook_kwargs["disable_for_tkns"], -kcodes:
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
                codebook_ids,
                torch.tensor(hook_kwargs["disable_codes"], device=codebook_ids.device),
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
    """Snap function with euclidean distance as similarity metric."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, codebook: torch.Tensor, kcodes, hook_kwargs):
        """Compute output of the snap function with the minimum euclidean distance as the similarity metric.

        Replaces each dimension vector of input with features from codebook
        having highest dot-product.

        Args:
        ----
            ctx: torch context used for efficiently storing tensors for backward pass.
            inputs: input data.
            codebook: codebook matrix. Shape: (num_features, hidden_dim_size).
            kcodes: number of codebook features to use for computing the output.
            hook_kwargs: dictionary of hook arguments.

        Returns: tuple of output of snap function and the IDs of closest codebook features.
        """
        logits = -torch.cdist(inputs, codebook, p=2)  # logits are negative distances
        if hook_kwargs["keep_k_codes"]:
            if hook_kwargs["disable_for_tkns"] == "all":
                logits[:, :, hook_kwargs["disable_codes"]] = float("-inf")
                _, codebook_ids = logits.topk(
                    kcodes + hook_kwargs["disable_topk"], dim=-1
                )
                codebook_ids = codebook_ids[:, :, -kcodes:]
            else:
                for code in hook_kwargs["disable_codes"]:
                    logits[:, hook_kwargs["disable_for_tkns"], code] = float("-inf")
                _, codebook_ids_all = logits.topk(
                    kcodes + hook_kwargs["disable_topk"], dim=-1
                )
                codebook_ids = codebook_ids_all[:, :, :kcodes]
                codebook_ids[:, hook_kwargs["disable_for_tkns"]] = codebook_ids_all[
                    :, hook_kwargs["disable_for_tkns"], -kcodes:
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
                codebook_ids,
                torch.tensor(hook_kwargs["disable_codes"], device=codebook_ids.device),
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


class CodebookLayer(nn.Module):
    """Codebook layer module."""

    def __init__(
        self,
        dim: int,
        num_codes: int,
        key: str,
        soft_snap: bool = False,
        snap_fn: Type[BaseSnapFunction] = EuclideanSnapFunction,
        hook_fn: Optional[Callable] = None,
        kmeans_init=False,
        kmeans_kwargs: Optional[Dict] = None,
        kcodes=1,
        **kwargs,
    ):
        """Create the codebook layer.

        Args:
        ----
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            key: key to identify the codebook in hook caches.
            soft_snap: whether to snap the input using softmax. Defaults to False.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            hook_fn: hook function apply to codebook ids.
            kmeans_init: whether to initialize the codebook with k-means of the data. Defaults to False.
            kmeans_kwargs: dictionary of arguments to pass to k-means embedding layer.
            kcodes: number of codebook features to use for computing the output.
            kwargs: additional arguments.
        """
        super().__init__()
        if kmeans_init:
            if kmeans_kwargs is None:
                kmeans_kwargs = {}
            self.codebook: nn.Embedding = KMeansEmbedding(
                num_embeddings=num_codes, embedding_dim=dim, **kmeans_kwargs
            )
        else:
            self.codebook = nn.Embedding(num_embeddings=num_codes, embedding_dim=dim)
        self.ln = torch.nn.LayerNorm(dim, eps=1e-05)
        self._num_codes = num_codes
        self.soft_snap = soft_snap
        self.snap_fn = snap_fn
        self.hook_fn = hook_fn
        self.key = key
        self.kcodes = kcodes

        # metrics
        self.counts = torch.zeros(num_codes, dtype=torch.long)
        self.reconstruction_mse = 0.0
        self.input_norm = 0.0
        self.output_norm = 0.0
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
        """Enable logging."""
        self.logging = True

    def disable_logging(self):
        """Disable logging."""
        self.logging = False

    def initialize_codebook(self):
        """Initialize the codebook using k-means.

        Args:
        ----
            data: data to use for initializing the codebook.
        """
        assert isinstance(self.codebook, KMeansEmbedding)
        self.codebook.initialize()

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
                self.ln(x),
                self.codebook.weight,
                self.kcodes,
                self.hook_kwargs,
            )  # type: ignore
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
                coeff: float = x.shape[0] * x.shape[1]
                coeff /= self.tokens_processed + x.shape[0] * x.shape[1]
                mse = torch.mean(((x - output) ** 2).sum(dim=-1), dim=None)
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
        """Set the hook kwargs."""
        self.hook_kwargs = {**self.hook_kwargs, **kwargs}
        assert all(
            k
            in [
                "disable_topk",
                "disable_codes",
                "disable_for_tkns",
                "keep_k_codes",
                "disable_sim_idx",
                "cosine",
            ]
            for k in self.hook_kwargs
        )

    def reset_hook_kwargs(self):
        """Reset the hook kwargs."""
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
        assert isinstance(self.codebook, KMeansEmbedding)
        self.codebook.load_data(data)

    def clear_data(self):
        """Clear the data for kmeans."""
        assert isinstance(self.codebook, KMeansEmbedding)
        self.codebook.clear_data()

    def initialize(self):
        """Initialize the codebook with kmeans."""
        assert isinstance(self.codebook, KMeansEmbedding)
        self.codebook.initialize()

    def partial_fit_codebook(self):
        """Update the codebook with the data."""
        assert isinstance(self.codebook, KMeansEmbedding)
        self.codebook.partial_fit()


class CompositionalCodebookLayer(nn.Module):
    """Module that applies distinct codebooks to chunks of input vectors."""

    def __init__(
        self,
        dim: int,
        num_codes: int,
        key: str,
        num_codebooks: int = 1,
        kmeans_init=False,
        soft_snap: bool = False,
        snap_fn: Type[BaseSnapFunction] = EuclideanSnapFunction,
        hook_fn: Optional[Callable] = None,
        kmeans_kwargs: Optional[Dict] = None,
        kcodes: int = 1,
    ):
        """Create the compositional codebook layer.

        Args:
        ----
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            key: key to identify the codebook in hook caches.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
            soft_snap: whether to snap the input using softmax. Defaults to False.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            hook_fn: hook function apply to codebook ids.
            kmeans_init: whether to initialize the codebook with k-means of the data. Defaults to False.
            kmeans_kwargs: dictionary of arguments to pass to k-means embedding layer.
            kcodes: number of codebook features to use for computing the output.
        """
        super().__init__()
        if dim % num_codebooks != 0:
            raise ValueError(
                "dim must be divisible by num_codebooks. Got dim: {}, num_codebooks: {}".format(
                    dim, num_codebooks
                )
            )
        self.num_codebooks = num_codebooks
        if kmeans_kwargs is None:
            kmeans_kwargs = {}
        seed = kmeans_kwargs.get("random_state", 0)
        self.codebook: Any = nn.ModuleList(
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
        """Enable logging for all the codebooks."""
        for codebook in self.codebook:
            codebook.enable_logging()

    def disable_logging(self):
        """Disable logging for all the codebooks."""
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
            assert (
                x.shape[2] == self.num_codebooks
            ), f"{x.shape}; num_cbs: {self.num_codebooks}"
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
        """Set the hook kwargs for the codebooks."""
        if head_idx is not None:
            if type(head_idx) == int:
                head_idx = [head_idx]
            for i in head_idx:
                self.codebook[i].set_hook_kwargs(**kwargs)
            return
        for codebook in self.codebook:
            codebook.set_hook_kwargs(**kwargs)

    def reset_hook_kwargs(self):
        """Reset the hook kwargs for the codebooks."""
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
        snap_fn: Type[BaseSnapFunction] = EuclideanSnapFunction,
        num_codebooks: int = 1,
        hook_fn: Optional[Callable] = None,
        kmeans_kwargs: Optional[Dict] = None,
    ):
        """Create the compositional codebook layer.

        Args:
        ----
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            key: key to identify the codebook in hook caches.ff
            num_codebooks: number of codebooks to use. Should divide `dim`. Defaults to 1.
            soft_snap: whether to snap the input using softmax. Defaults to False.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            hook_fn: hook function apply to codebook ids.
            kmeans_init: whether to initialize the codebook with k-means of the data. Defaults to False.
            kmeans_kwargs: dictionary of arguments to pass to k-means embedding layer.
            kcodes: number of codebook features to use for computing the output.
        """
        super().__init__()
        if num_codes % num_codebooks != 0:
            raise ValueError(
                "num_codes must be divisible by num_codebooks."
                f" Got num_codes: {num_codes}, num_codebooks: {num_codebooks}"
            )

        if kmeans_kwargs is None:
            kmeans_kwargs = {}
        seed = kmeans_kwargs.get("random_state", 0)
        self.codebook: Any = nn.ModuleList(
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


class VQCodebookLayer(nn.Module):
    """Codebook Layer that uses vqtorch's implementation of codebooks."""

    def __init__(
        self,
        dim: int,
        num_codes: int,
        key: str,
        soft_snap: bool = False,
        snap_fn: Type[BaseSnapFunction] = EuclideanSnapFunction,
        hook_fn: Optional[Callable] = None,
        kmeans_init=False,
        kmeans_kwargs: Optional[Dict] = None,
        kcodes: int = 1,
        beta: float = 0.95,
        sync_nu: float = 0.0,
        affine_lr: float = 0.0,
        affine_groups: int = 1,
        replace_freq: int = 0,
        optimizer_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """Create the vqtorch codebook layer.

        Args:
        ----
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            key: key to identify the codebook in hook caches.
            soft_snap: whether to snap the input using softmax. Defaults to False.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            hook_fn: hook function apply to codebook ids.
            kmeans_init: whether to initialize the codebook with k-means of the data. Defaults to False.
            kmeans_kwargs: dictionary of arguments to pass to k-means embedding layer.
            kcodes: number of codebook features to use for computing the output.
            beta: commitment loss weighting
            sync_nu: sync loss weighting
            affine_lr: learning rate for affine transform
            affine_groups: number of affine parameter groups
            replace_freq: frequency to replace dead codes
            optimizer_kwargs: arguments for the optimizer
            kwargs: additional arguments for _VQBaseLayer

        """
        super().__init__()
        self.dim = dim
        self.num_codes = num_codes
        self.key = key
        self.soft_snap = soft_snap
        if self.soft_snap:
            raise NotImplementedError("Soft snap not implemented yet.")
        self.snap_fn = snap_fn
        self.hook_fn = hook_fn
        self.kmeans_init = kmeans_init
        self.kmeans_kwargs = kmeans_kwargs
        self.kcodes = kcodes
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        def inplace_optimizer(*_args, **_kwargs):
            return torch.optim.SGD(*_args, **_kwargs, **optimizer_kwargs)

        if "num_codebooks" in kwargs:
            kwargs.pop("num_codebooks")
        self.vq_layer = vqtorch_nn.VectorQuant(
            feature_size=dim,
            num_codes=num_codes,
            beta=beta,
            sync_nu=sync_nu,
            affine_lr=affine_lr,
            affine_groups=affine_groups,
            replace_freq=replace_freq,
            inplace_optimizer=inplace_optimizer,  # type: ignore
            dim=-1,
            **kwargs,
        )

        # metrics
        self.counts = torch.zeros(num_codes, dtype=torch.long)
        self.reconstruction_mse = 0.0
        self.input_norm = 0.0
        self.output_norm = 0.0
        self.tokens_processed = 0
        self.hook_codebook_ids = HookPoint()
        self.logging = True

    @property
    def codebook(self) -> nn.Embedding:
        """Return the codebook."""
        return self.vq_layer.codebook

    @property
    def active_codes(self):
        """Return the number of active codes."""
        return torch.sum(self.counts != 0).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Snaps activations to elements in the codebook.

        Args:
        ----
            x: input tensor of shape: (batch_size, seq_len, dim).

        Returns: output with the feature vectors replaced using the codebook.
        """
        assert len(x.shape) == 3
        output, return_dict = self.vq_layer(x)
        codebook_ids = return_dict["q"]
        if self.logging:
            with torch.no_grad():
                self.codes_triggered = torch.unique(
                    codebook_ids.cpu(), sorted=False, return_counts=False
                )
                self.counts[self.codes_triggered] += 1
            coeff: float = x.shape[0] * x.shape[1]
            coeff /= self.tokens_processed + x.shape[0] * x.shape[1]
            mse = torch.mean(((x - output) ** 2).sum(dim=-1), dim=None)
            self.reconstruction_mse += coeff * (mse.item() - self.reconstruction_mse)
            self.input_norm += coeff * (
                torch.norm(x, dim=-1).mean().item() - self.input_norm
            )
            self.output_norm += coeff * (
                torch.norm(output, dim=-1).mean().item() - self.output_norm
            )
            self.tokens_processed += x.shape[0] * x.shape[1]

        if self.hook_fn is not None:
            self.hook_fn(self.key, codebook_ids.cpu().numpy())

        return output

    def enable_logging(self):
        """Enable logging."""
        self.logging = True

    def disable_logging(self):
        """Disable logging."""
        self.logging = False

    def get_most_used_code(self):
        """Return the most used code."""
        return self.vq_layer.get_codebook()[self.counts.argmax()].detach().cpu().numpy()

    def set_hook_fn(self, hook_fn: Callable):
        """Set the hook function.

        Args:
        ----
            hook_fn: hook function to use.
        """
        self.hook_fn = hook_fn

    def reset_metrics(self):
        """Reset the counts of the codebook features."""
        self.counts.zero_()
        self.reconstruction_mse = 0
        self.tokens_processed = 0
        self.input_norm = 0
        self.output_norm = 0

    def avg_norm(self):
        """Return the average norm of the codebook features."""
        # TODO: check for affine transformations
        return self.codebook.weight.norm(p=2, dim=1).mean().item()

    def max_norm(self):
        """Return the maximum norm of the codebook features."""
        return self.codebook.weight.norm(p=2, dim=1).max().item()

    def most_common_counts(self):
        """Return the most common codebook feature counts."""
        return self.counts.sort()[0].cpu().numpy()


class CompositionalVQCodebookLayer(nn.Module):
    """Codebook Layer that uses vqtorch's implementation of codebooks."""

    def __init__(
        self,
        dim: int,
        num_codes: int,
        key: str,
        num_codebooks: int = 1,
        soft_snap: bool = False,
        snap_fn: Type[BaseSnapFunction] = EuclideanSnapFunction,
        hook_fn: Optional[Callable] = None,
        kmeans_init=False,
        kmeans_kwargs: Optional[Dict] = None,
        kcodes: int = 1,
        beta: float = 0.95,
        sync_nu: float = 0.0,
        affine_lr: float = 0.0,
        affine_groups: int = 1,
        replace_freq: int = 0,
        optimizer_kwargs: Optional[Dict] = None,
        share: bool = False,
        **kwargs,
    ):
        """Create the vqtorch codebook layer.

        Args:
        ----
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            num_codebooks: number of codebooks to use. Defaults to 1.
            key: key to identify the codebook in hook caches.
            soft_snap: whether to snap the input using softmax. Defaults to False.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            hook_fn: hook function apply to codebook ids.
            kmeans_init: whether to initialize the codebook with k-means of the data. Defaults to False.
            kmeans_kwargs: dictionary of arguments to pass to k-means embedding layer.
            kcodes: number of codebook features to use for computing the output.
            beta: commitment loss weighting.
            sync_nu: sync loss weighting.
            affine_lr: learning rate for affine transform.
            affine_groups: number of affine parameter groups.
            replace_freq: frequency to replace dead codes.
            optimizer_kwargs: arguments for the optimizer.
            share: whether to share the codebooks across layers.
            kwargs: additional arguments for _VQBaseLayer

        """
        super().__init__()
        self.dim = dim
        self.num_codes = num_codes
        self.num_codebooks = num_codebooks
        self.key = key
        self.soft_snap = soft_snap
        if self.soft_snap:
            raise NotImplementedError("Soft snap not implemented yet.")
        self.snap_fn = snap_fn
        self.hook_fn = hook_fn
        self.kmeans_init = kmeans_init
        self.kmeans_kwargs = kmeans_kwargs
        self.kcodes = kcodes
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        def inplace_optimizer(*_args, **_kwargs):
            return torch.optim.SGD(*_args, **_kwargs, **optimizer_kwargs)

        if "num_codebooks" in kwargs:
            kwargs.pop("num_codebooks")
        self.vq_layer = vqtorch_nn.GroupVectorQuant(
            feature_size=dim,
            num_codes=num_codebooks
            * num_codes,  # GVQ expects total number of codes across groups
            groups=num_codebooks,
            share=share,
            beta=beta,
            sync_nu=sync_nu,
            affine_lr=affine_lr,
            affine_groups=affine_groups,
            replace_freq=replace_freq,
            inplace_optimizer=inplace_optimizer,  # type: ignore
            dim=-1,
            **kwargs,
        )

        # metrics
        self.counts = torch.zeros(num_codebooks * num_codes, dtype=torch.long)
        self.reconstruction_mse = 0.0
        self.input_norm = 0.0
        self.output_norm = 0.0
        self.tokens_processed = 0
        self.hook_codebook_ids = HookPoint()
        self.logging = True

    @property
    def codebook(self) -> nn.Embedding:
        """Return the codebook."""
        return self.vq_layer.codebook

    @property
    def active_codes(self):
        """Return the number of active codes."""
        return torch.sum(self.counts != 0).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Snaps activations to elements in the codebook.

        Args:
        ----
            x: input tensor of shape: (batch_size, seq_len, dim).

        Returns: output with the feature vectors replaced using the codebook.
        """
        assert len(x.shape) == 3
        output, return_dict = self.vq_layer(x)
        codebook_ids = return_dict["q"]
        if self.logging:
            with torch.no_grad():
                self.codes_triggered = torch.unique(
                    codebook_ids.cpu(), sorted=False, return_counts=False
                )
                self.counts[self.codes_triggered] += 1
            coeff: float = x.shape[0] * x.shape[1]
            coeff /= self.tokens_processed + x.shape[0] * x.shape[1]
            mse = torch.mean(((x - output) ** 2).sum(dim=-1), dim=None)
            self.reconstruction_mse += coeff * (mse.item() - self.reconstruction_mse)
            self.input_norm += coeff * (
                torch.norm(x, dim=-1).mean().item() - self.input_norm
            )
            self.output_norm += coeff * (
                torch.norm(output, dim=-1).mean().item() - self.output_norm
            )
            self.tokens_processed += x.shape[0] * x.shape[1]

        if self.hook_fn is not None:
            self.hook_fn(self.key, codebook_ids.cpu().numpy())

        return output

    def enable_logging(self):
        """Enable logging."""
        self.logging = True

    def disable_logging(self):
        """Disable logging."""
        self.logging = False

    def get_most_used_code(self):
        """Return the most used code."""
        return self.vq_layer.get_codebook()[self.counts.argmax()].detach().cpu().numpy()

    def set_hook_fn(self, hook_fn: Callable):
        """Set the hook function.

        Args:
        ----
            hook_fn: hook function to use.
        """
        self.hook_fn = hook_fn

    def reset_metrics(self):
        """Reset the counts of the codebook features."""
        self.counts.zero_()
        self.reconstruction_mse = 0
        self.tokens_processed = 0
        self.input_norm = 0
        self.output_norm = 0

    def avg_norm(self):
        """Return the average norm of the codebook features."""
        # TODO: check for affine transformations
        return self.codebook.weight.norm(p=2, dim=1).mean().item()

    def max_norm(self):
        """Return the maximum norm of the codebook features."""
        return self.codebook.weight.norm(p=2, dim=1).max().item()

    def most_common_counts(self):
        """Return the most common codebook feature counts."""
        return self.counts.sort()[0].cpu().numpy()


class CodebookWrapper(nn.Module, abc.ABC):
    """Abstract class to wraps a nn module by applying codebooks on the output of the layer."""

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
        snap_fn: Type[BaseSnapFunction] = EuclideanSnapFunction,
        num_codebooks: int = 1,
        hook_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """Create the transformer layer wrapped with the codebook.

        Args:
        ----
            module_layer: module layer to wrap codebook on.
            codebook_cls: codebook class to use. Can be either `CodebookLayer` (default),
                `CompositionalCodebookLayer` or `GroupedCodebookLayer`.
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            key: key to identify the codebook in hook caches.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
            hook_fn: hook function to apply to codebook ids. Defaults to None.
            kwargs: additional keyword arguments to pass to the codebook.
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
        """Forward pass of the wrapped module."""
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
        snap_fn: Type[BaseSnapFunction] = EuclideanSnapFunction,
        num_codebooks: int = 1,
        hook_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """Create the transformer layer wrapped with the codebook.

        Args:
        ----
            module_layer: module layer to wrap codebook on.
            codebook_cls: codebook class to use. Can be either `CodebookLayer` (default),
                `CompositionalCodebookLayer` or `GroupedCodebookLayer`.
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            key: key to identify the codebook in hook caches.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
            hook_fn: hook function to apply to codebook ids. Defaults to None.
            kwargs: additional keyword arguments to pass to the codebook.
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
        snap_fn: Type[BaseSnapFunction] = EuclideanSnapFunction,
        num_codebooks: int = 1,
        hook_fn: Optional[Callable] = None,
        **kwargs,
    ):
        """Create the MLP layer wrapped with the codebook.

        Args:
        ----
            module_layer: module layer to wrap codebook on.
            codebook_cls: codebook class to use. Can be either `CodebookLayer` (default),
                `CompositionalCodebookLayer` or `GroupedCodebookLayer`.
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            key: key to identify the codebook in hook caches.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
            hook_fn: hook function to apply to codebook ids. Defaults to None.
            kwargs: additional keyword arguments to pass to the codebook.
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
    """Configuration class to store the configuration of a `CodebookModel`."""

    model_type = "codebook"

    def __init__(
        self,
        codebook_type: Union[str, Sequence] = "vanilla",
        num_codes: int = 100,
        num_codebooks: Union[int, Sequence] = 1,
        layers_to_snap: Sequence = (),
        similarity_metric: str = "inner_product",
        codebook_at: Union[str, Sequence] = "mlp",
        loss: str = "base",
        k_codebook: Union[int, Sequence] = 1,
        kmeans_init: bool = False,
        kmeans_init_examples: int = 1000,
        kmeans_path: Optional[str] = None,
        kmeans_kwargs: Optional[Dict] = None,
        codebook_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Create the config for the codebook model.

        Args:
        ----
            codebook_type: type of codebook to use. Can be either 'vanilla' (default, uses `CodebookLayer`),
                'compositional', 'grouped', or 'vqtorch'.
            num_codes: number of codebook features to have.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
            layers_to_snap: Index of transformer layers in the model on which codebook to apply.
                Defaults to []. Can contain negative numbers to index from the last layers.
            similarity_metric: similarity metric to use. Can be either 'euclidean' (default) or 'inner_product'.
            codebook_at: where to apply codebook. Can be either 'mlp' (default) or 'transformer_block'.
            loss: whether to use the loss used in VQVAE paper or the CLM loss.
            k_codebook: number of nearest neighbors in codebook snapping.
            kmeans_init: whether to initialize codebook with kmeans.
            kmeans_init_examples: number of examples to use for kmeans initialization.
            kmeans_path: path to load or save the kmeans embeddings.
            kmeans_kwargs: additional keyword arguments to pass to kmeans.
            codebook_kwargs: additional keyword arguments to pass to the codebook layer.
            kwargs: additional keyword arguments to pass to the config.
        """
        super().__init__(**kwargs)
        if type(codebook_type) == str:
            codebook_type = [codebook_type]
        if type(num_codebooks) == int:
            num_codebooks = [num_codebooks] * len(codebook_type)
        if type(k_codebook) == int:
            k_codebook = [k_codebook] * len(codebook_type)
        for i in range(len(codebook_type)):
            if codebook_type[i] not in [
                "vanilla",
                "compositional",
                "grouped",
                "vqtorch",
                "comp-vqtorch",
            ]:
                raise ValueError(f"Invalid codebook type {codebook_type[i]}")
            if codebook_type[i] == "vanilla" and num_codebooks[i] != 1:  # type: ignore
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
        if codebook_kwargs is None:
            codebook_kwargs = {}
        self.codebook_kwargs = codebook_kwargs


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
        self.codebook_cls: List = []
        self.model: Any = model
        self.logging = True
        self.model_params = list(model.parameters())
        self.codebook_params = []
        self.all_codebook_wrappers = {}
        self.init_codebook_classes()
        self.set_codebook_args()
        self.add_codebooks()
        # override the forward method
        self.forward = self.model.forward

    def __setattr__(self, name: str, value: Any) -> None:
        """If the model is set, override the forward method using the new model.

        Required when loading a codebook model using `from_pretrained` method.
        """
        super().__setattr__(name, value)
        if name == "model":
            self.forward = self.model.forward

    # labels is needed in the signature so that transformers.trainer can return loss
    def forward(self, *args, labels: Optional[torch.LongTensor] = None, **kwargs):
        """Raises an error if this method is called."""
        raise RuntimeError(
            "This shouldn't get executed as forward is overridden in init."
        )

    def init_codebook_classes(self):
        """Initialize the codebook classes based on the `codebook_type` configuration."""
        for cb_type in self.config.codebook_type:
            if cb_type == "vanilla":
                self.codebook_cls.append(CodebookLayer)
            elif cb_type == "compositional":
                self.codebook_cls.append(CompositionalCodebookLayer)
            elif cb_type == "grouped":
                self.codebook_cls.append(GroupedCodebookLayer)
            elif cb_type == "vqtorch":
                self.codebook_cls.append(VQCodebookLayer)
            elif cb_type == "comp-vqtorch":
                self.codebook_cls.append(CompositionalVQCodebookLayer)

    def set_codebook_args(self):
        """Set the number of codebooks based on the `num_codebooks` configuration."""
        for i in range(len(self.config.num_codebooks)):
            if self.config.num_codebooks[i] == -1:
                self.config.num_codebooks[i] = self.num_heads
        num_layers = self.num_layers()
        if self.config.layers_to_snap == "all":
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
        """Gets attributes from the wrapped model if not found in the codebook model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def generate(self, *args, **kwargs):
        """Generate output from the wrapped model."""
        return self.model.generate(*args, **kwargs)

    @property
    def all_codebooks(self):
        """Returns a dictionary of layer idx to all codebooks in that layer."""
        return {
            k: [v.codebook_layer for v in vl]
            for k, vl in self.all_codebook_wrappers.items()
        }

    def add_codebooks(self):
        """Adds codebooks for the layers that are to be snapped."""
        CODEBOOK_METHODS = {
            "transformer_block": "codebook_at_transformer",
            "mlp": "codebook_at_mlp",
            "mlp_mid": "codebook_at_mlp_mid",
            "qkv": "codebook_at_qkv",
            "attention": "codebook_at_attention",
            "preproj_attention": "codebook_at_preprojection_attn",
            "attention_and_mlp": "codebook_at_attention_plus_mlp",
        }
        layers = self.layers()
        for i in range(len(layers)):
            if i in self.config.layers_to_snap:
                codebooks_in_layer = []
                for i_cb, cb_at in enumerate(self.config.codebook_at):
                    method_name = CODEBOOK_METHODS.get(cb_at)
                    if method_name is not None:
                        method = getattr(self, method_name)
                        method(layers, i, codebooks_in_layer, i_cb)

                if not codebooks_in_layer:
                    raise ValueError(
                        f"Invalid value for `codebook_at`: {self.config.codebook_at}."
                    )
                self.all_codebook_wrappers[i] = codebooks_in_layer

    def codebook_at_transformer(self, layers, i, codebooks_in_layer, i_cb):
        """Adds codebook at the transformer block."""
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
            **self.config.codebook_kwargs,
        )
        self.codebook_params += list(
            layers[i].codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(layers[i])

    def codebook_at_mlp(self, layers, i, codebooks_in_layer, i_cb):
        """Adds codebook at output of the MLP layer."""
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
            **self.config.codebook_kwargs,
        )
        layers[i].__setattr__(self.mlp_key, wrapped_mlp)
        self.codebook_params += list(
            wrapped_mlp.codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(wrapped_mlp)

    def codebook_at_mlp_mid(self, layers, i, codebooks_in_layer, i_cb):
        """Adds codebook at the hidden layer of MLP."""
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
            **self.config.codebook_kwargs,
        )
        mlp.__setattr__(self.mlp_mid_key, wrapped_hidden_layer)
        self.codebook_params += list(
            wrapped_hidden_layer.codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(wrapped_hidden_layer)

    def codebook_at_qkv(self, layers, i, codebooks_in_layer, i_cb):
        """Adds a separate codebook at each of the q, k, v layers."""
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
            **self.config.codebook_kwargs,
        )
        attn.__setattr__(self.qkv_key, wrapped_hidden_layer)
        self.codebook_params += list(
            wrapped_hidden_layer.codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(wrapped_hidden_layer)

    def codebook_at_attention(self, layers, i, codebooks_in_layer, i_cb):
        """Adds codebook at the output of the attention layer."""
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
            **self.config.codebook_kwargs,
        )
        layers[i].__setattr__(self.attention_key, wrapped_attn)
        self.codebook_params += list(
            wrapped_attn.codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(wrapped_attn)

    def codebook_at_preprojection_attn(self, layers, i, codebooks_in_layer, i_cb):
        """Adds codebook at the attention layer before the output is projected to the residual stream."""
        codebook = self.codebook_cls[i_cb](
            dim=self.d_model,
            num_codes=self.config.num_codes,
            key=f"layer{i}_attn_preproj",
            snap_fn=self.snap_fn,
            num_codebooks=self.config.num_codebooks[i_cb],
            kmeans_init=self.config.kmeans_init,
            kmeans_kwargs=self.config.kmeans_kwargs,
            kcodes=self.config.k_codebook[i_cb],
            **self.config.codebook_kwargs,
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
        """Adds codebook on the summed output of attention and MLP layers."""
        codebook = self.codebook_cls[i_cb](
            dim=self.d_model,
            num_codes=self.config.num_codes,
            key=f"layer{i}_attn+mlp",
            snap_fn=self.snap_fn,
            num_codebooks=self.config.num_codebooks[i_cb],
            kmeans_init=self.config.kmeans_init,
            kmeans_kwargs=self.config.kmeans_kwargs,
            kcodes=self.config.k_codebook[i_cb],
            **self.config.codebook_kwargs,
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
        """Sets the hook kwargs for the codebook layers in `idx`.

        If `idx` is None, sets the hook kwargs for all codebook layers.
        """
        if idx is not None:
            if type(idx) == int:
                idx = [idx]
            for i in idx:
                layers = self.all_codebooks[i]
                for layer in layers:
                    layer.set_hook_kwargs(**kwargs)
            return
        for _i, layers in self.all_codebooks.items():
            for layer in layers:
                layer.set_hook_kwargs(**kwargs)

    def reset_hook_kwargs(self, idx=None):
        """Resets the hook kwargs for the codebook layers in `idx`."""
        if idx is not None:
            for layer in list(self.all_codebooks.values())[idx]:
                layer.reset_hook_kwargs()
            return
        for _i, layers in self.all_codebooks.items():
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
        for _i, layers in self.all_codebooks.items():
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
        """Enables logging for all the codebooks."""
        self.logging = True
        for codebooks in self.all_codebooks.values():
            for codebook in codebooks:
                codebook.enable_logging()

    def disable_logging(self):
        """Disables logging for all the codebooks."""
        self.logging = False
        for codebooks in self.all_codebooks.values():
            for codebook in codebooks:
                codebook.disable_logging()

    @property
    @abc.abstractmethod
    def pre_projection_attn_codebook_cls(self) -> Any:
        """Returns the class of pre projection attention codebook."""
        pass

    @property
    @abc.abstractmethod
    def pre_residual_codebook_cls(self) -> Any:
        """Returns the class of pre residual codebook."""
        pass

    @abc.abstractmethod
    def itermediate_size(self) -> int:
        """Returns the intermediate size of the model."""
        pass

    @property
    @abc.abstractmethod
    def mlp_mid_key(self) -> str:
        """Returns the key of layer in MLP layer where codebook is to be applied."""
        pass

    @property
    @abc.abstractmethod
    def d_model(self) -> int:
        """Returns the dimension of the model."""
        pass

    @abc.abstractmethod
    def layers(self) -> Sequence[nn.Module]:
        """Returns the list of transformer layers of the model."""
        pass

    @abc.abstractmethod
    def num_layers(self) -> int:
        """Returns the number of transformer layers in the model."""
        pass

    @abc.abstractmethod
    def base_model_cfg(self):
        """Returns the base model config."""
        pass


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

    def layers(self):
        """Returns the list of transformer layers of the model."""
        return self.model.transformer.h  # type: ignore

    def num_layers(self):
        """Returns the number of transformer layers in the model."""
        return self.model.config.n_layer  # type: ignore

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
        """Returns the attribute name used for qkv in the model."""
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
        """Returns the attribute name used for qkv layer in the model."""
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
            config: config for the codebook model.
            model: HookedTransformer model to apply codebooks to.
            base_model_config: config for the base model on which HookedTransformer is applied.
        """
        # get config key values from original class
        for k, v in base_model_config.__dict__.items():
            if k not in model.cfg.__dict__:
                model.cfg.__setattr__(k, v)
        for k1, k2 in base_model_config.attribute_map.items():
            if k1 not in model.cfg.__dict__:
                model.cfg.__setattr__(k1, base_model_config.__getattribute__(k2))
        super().__init__(
            config=config,
            model=model,
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
        """Returns the device of the model."""
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
    else:
        raise ValueError(
            f"Model type {model.config.model_type} not supported with codebooks."
        )


def convert_to_hooked_model(model_path, orig_cb_model, hooked_kwargs=None):
    """Wraps a hooked tranformer model with codebooks."""
    if hooked_kwargs is None:
        hooked_kwargs = {}
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


def convert_to_hooked_model_for_toy(
    model_path,
    orig_cb_model,
    config,
    hooked_kwargs=None,
):
    """Wraps a hooked tranformer model with codebooks."""
    hooked_config = loading.convert_hf_model_config(model_path, config)
    model = transformer_lens.HookedTransformer(hooked_config)
    if hooked_kwargs is None:
        hooked_kwargs = {}
    if "device" in hooked_kwargs:
        hooked_kwargs.pop("device")
    state_dict = convert_state_dict(orig_cb_model.model, model.cfg)  # type: ignore
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
    """Converts a state_dict from a HuggingFace model to a state_dict compatible with HookedTransformer."""
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
        raise ValueError(f"Unknown architecture {cfg.original_architecture}")
