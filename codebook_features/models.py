"""Model related classes."""

import abc
import os
import re
from typing import Any, Callable, Mapping, Optional, Sequence, Type, Union

import numpy as np
import torch
import transformer_lens
import transformers
from sklearn import cluster as sklearn_cluster
from torch import nn
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from codebook_features import mod_model_classes, tl_mods, utils

try:
    import faiss
    import faiss.contrib.torch_utils

    try:
        RES = faiss.StandardGpuResources()
    except (RuntimeError, AttributeError):
        RES = None
except ImportError:
    faiss = None


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
        self.kmeans = sklearn_cluster.MiniBatchKMeans(n_clusters=num_embeddings, **kmeans_kwargs)

    def load_data(self, data: torch.Tensor):
        """Load a batch of data.

        Args:
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
            k: number of cluster centers for K-Means.
        """
        self.weight.data = torch.from_numpy(self.kmeans.cluster_centers_).to(self.weight.device)
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
            with torch.enable_grad():
                mse_loss = torch.mean(((outputs - inputs) ** 2).sum(dim=-1))
                grad_codebook_mse = torch.autograd.grad(mse_loss, codebook, retain_graph=True)[0]
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
                grad_codebook_mse = torch.autograd.grad(mse_loss, codebook, retain_graph=True)[0]
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
            ctx: torch context used for efficiently storing tensors for backward pass.
            inputs: input data.
            codebook: codebook matrix. Shape: (num_features, hidden_dim_size).
            kcodes: number of codebook features to use for computing the output.
            hook_kwargs: dictionary of hook arguments.

        Returns: tuple of output of snap function and the IDs of closest codebook features.
        """
        # normalize codebook
        cb_norm = torch.linalg.vector_norm(codebook, dim=-1).reshape(1, 1, -1)
        logits = torch.matmul(inputs, codebook.T) / cb_norm
        if hook_kwargs["cosine"]:
            logits = logits / torch.norm(inputs, dim=-1, keepdim=True)
        if hook_kwargs["keep_k_codes"]:
            if hook_kwargs["disable_for_tkns"] == "all":
                logits[:, :, hook_kwargs["disable_codes"]] = float("-inf")
                _, codebook_ids = logits.topk(kcodes + hook_kwargs["disable_topk"], dim=-1)
                codebook_ids = codebook_ids[:, :, -kcodes:]
            else:
                for code in hook_kwargs["disable_codes"]:
                    logits[:, hook_kwargs["disable_for_tkns"], code] = float("-inf")
                _, codebook_ids_all = logits.topk(kcodes + hook_kwargs["disable_topk"], dim=-1)
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
                unblock_tkns = torch.zeros(codebook_ids.shape[1], dtype=torch.bool, device=codebook_ids.device)
            else:
                unblock_tkns = torch.ones(codebook_ids.shape[1], dtype=torch.bool, device=codebook_ids.device)
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
                _, codebook_ids = logits.topk(kcodes + hook_kwargs["disable_topk"], dim=-1)
                codebook_ids = codebook_ids[:, :, -kcodes:]
            else:
                for code in hook_kwargs["disable_codes"]:
                    logits[:, hook_kwargs["disable_for_tkns"], code] = float("-inf")
                _, codebook_ids_all = logits.topk(kcodes + hook_kwargs["disable_topk"], dim=-1)
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
                unblock_tkns = torch.zeros(codebook_ids.shape[1], dtype=torch.bool, device=codebook_ids.device)
            else:
                unblock_tkns = torch.ones(codebook_ids.shape[1], dtype=torch.bool, device=codebook_ids.device)
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


class FaissSnapFunction(BaseSnapFunction):
    """Snap function using the faiss library."""

    @staticmethod
    def forward(
        ctx,
        codebook_idx,
        inputs: torch.Tensor,
        codebook: torch.Tensor,
        kcodes: int,
        hook_kwargs: Mapping[str, Any],
    ) -> torch.Tensor:
        """Compute output of the snap function using the faiss library.

        Replaces each dimension vector of input with the closest features from codebook.
        Note that applying hooks with `hook_kwargs` is not supported with the faiss snap function.

        Args:
            ctx: torch context used for efficiently storing tensors for backward pass.
            inputs: input data.
            codebook: codebook matrix. Shape: (num_features, hidden_dim_size).
            codebook_idx: faiss index of the codebook.
            kcodes: number of codebook features to use for computing the output.
            hook_kwargs: dictionary of hook arguments.

        Returns: tuple of output of snap function and the IDs of closest codebook features.
        """
        # assumes normalized codebooks
        logits, codebook_ids = codebook_idx.search(inputs.reshape(-1, inputs.shape[-1]), kcodes)
        logits = logits.reshape(*(inputs.shape[:-1]), -1)
        codebook_ids = codebook_ids.reshape(*(inputs.shape[:-1]), -1)
        if hook_kwargs["cosine"]:
            logits = logits / torch.norm(inputs, dim=-1, keepdim=True)
        if hook_kwargs["disable_codes"] or hook_kwargs["disable_topk"]:
            raise NotImplementedError("Disabling codes not implemented for faiss.")
        outputs = torch.nn.functional.embedding(codebook_ids, codebook)
        outputs = outputs.sum(dim=-2) / kcodes
        return outputs, codebook_ids

    @staticmethod
    def backward(ctx, grad_outputs, grad_codebook_ids):
        """Backward pass for the snap function."""
        raise NotImplementedError("Backward pass not implemented for faiss.")


class CodebookLayer(nn.Module):
    """Codebook layer module."""

    def __init__(
        self,
        dim: int,
        num_codes: int,
        key: str,
        snap: bool = True,
        soft_snap: bool = False,
        snap_fn: Type[BaseSnapFunction] = EuclideanSnapFunction,
        hook_fn: Optional[Callable] = None,
        kmeans_init=False,
        kmeans_kwargs: Optional[Mapping] = None,
        kcodes: int = 1,
        replace_after_steps: int = 0,
        replace_rho: float = 0.0,
        **kwargs,
    ):
        """Create the codebook layer.

        Args:
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            key: key to identify the codebook in hook caches.
            snap: whether to snap (enable) codebooks. Defaults to True.
            soft_snap: whether to snap the input using softmax. Defaults to False.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            hook_fn: hook function apply to codebook ids.
            kmeans_init: whether to initialize the codebook with k-means of the data. Defaults to False.
            kmeans_kwargs: dictionary of arguments to pass to k-means embedding layer.
            kcodes: number of codebook features to use for computing the output.
            replace_after_steps: number of steps after which to replace a dead code.
            replace_rho: magnitude of random noise to add to a replaced code.
            kwargs: additional arguments.
        """
        super().__init__()
        if kmeans_init:
            if kmeans_kwargs is None:
                kmeans_kwargs = {}
            self.codebook: nn.Embedding = KMeansEmbedding(num_embeddings=num_codes, embedding_dim=dim, **kmeans_kwargs)
        else:
            self.codebook = nn.Embedding(num_embeddings=num_codes, embedding_dim=dim)
        self.ln = torch.nn.LayerNorm(dim, eps=1e-05)
        self._num_codes = num_codes
        self._snap = snap
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
        self.replace_after_steps = replace_after_steps
        self.replace_rho = replace_rho
        self.steps_since_replacement = 0

        self._store_data = False

    def normalize_L2(self):
        """Normalize the codebook features."""
        self.codebook.weight.data = torch.nn.functional.normalize(
            self.codebook.weight.data,
            p=2,
            dim=-1,
        )

    def use_faiss(self, device=None):
        """Use faiss for snap function."""
        self.snap_fn = FaissSnapFunction
        index_cls = faiss.IndexFlatIP if self.snap_fn == InnerProductSnapFunction else faiss.IndexFlatL2
        self.faiss_index = index_cls(self.codebook.weight.shape[-1])
        device = device or self.codebook.weight.device.type
        if device != "cpu":
            assert RES is not None, "GPU faiss not available."
            self.faiss_index = faiss.index_cpu_to_gpu(RES, 0, self.faiss_index)

        normalized_codebook = self.codebook.weight.detach().cpu()
        normalized_codebook = torch.nn.functional.normalize(normalized_codebook, p=2, dim=-1)
        self.faiss_index.add(normalized_codebook)

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

    def enable_codebook(self):
        """Enable the use of codebook."""
        self._snap = True

    def disable_codebook(self):
        """Disable the use of codebook. This passes the input as is."""
        self._snap = False

    def get_most_used_code(self):
        """Return the most used code."""
        return self.codebook.weight[self.counts.argmax()].detach().cpu().numpy()

    def set_hook_fn(self, hook_fn: Callable):
        """Set the hook function.

        Args:
            hook_fn: hook function to use.
        """
        self.hook_fn = hook_fn

    def forward(self, x: torch.Tensor):
        """Snaps activations to elements in the codebook.

        Args:
            x: input tensor of shape: (batch_size, seq_len, dim).

        Returns: output with the feature vectors replaced using the codebook.
        """
        assert len(x.shape) == 3  # (batch_size, seq_len, dim)
        if self.soft_snap:
            raise NotImplementedError("Soft snap not implemented.")
        if not self._snap:
            return x
        normalized_input = self.ln(x)
        if self._store_data:
            self.codebook.load_data(normalized_input)
            # store data is used for kmeans initialization and so
            # we don't want to snap the data and hence return here.
            return x
        else:
            if self.snap_fn == FaissSnapFunction:
                norm_output, codebook_ids = self.snap_fn.apply(
                    self.faiss_index,
                    normalized_input,
                    self.codebook.weight,
                    self.kcodes,
                    self.hook_kwargs,
                )
            else:
                norm_output, codebook_ids = self.snap_fn.apply(
                    normalized_input,
                    self.codebook.weight,
                    self.kcodes,
                    self.hook_kwargs,
                )  # type: ignore
            output = norm_output

        if not self.training:
            # this hook is used to modify/block activated codes during inference
            old_codebook_ids = codebook_ids.clone()
            codebook_ids = self.hook_codebook_ids(codebook_ids)
            block_idx = torch.isin(codebook_ids, -1)
            if torch.any(codebook_ids != old_codebook_ids) or torch.any(block_idx):
                # TODO: clean this.
                # change -1 to 0 and zero ablate the blocked codes before computing mean
                codebook_ids[block_idx] = 0
                norm_output = self.codebook(codebook_ids)
                norm_output[block_idx] = 0
                norm_output = norm_output.mean(dim=-2)
                output = norm_output
                codebook_ids[block_idx] = -1

        self.update_metrics(codebook_ids.cpu(), normalized_input, norm_output)

        if self.hook_fn is not None:
            self.hook_fn(self.key, codebook_ids.cpu().numpy())
        return output

    def update_metrics(
        self,
        codebook_ids: torch.Tensor,
        normalized_input: torch.Tensor,
        norm_output: torch.Tensor,
    ):
        """Update the logging metrics for the codebook."""
        if not self.logging:
            return
        self.codes_triggered = torch.unique(codebook_ids, sorted=False, return_counts=False)
        self.counts[self.codes_triggered] += 1
        num_tokens: int = codebook_ids.shape[0] * codebook_ids.shape[1]
        coeff = num_tokens / (self.tokens_processed + num_tokens)
        mse = torch.mean(((normalized_input - norm_output) ** 2).sum(dim=-1), dim=None)
        self.reconstruction_mse += coeff * (mse.item() - self.reconstruction_mse)
        self.input_norm += coeff * (torch.norm(normalized_input, dim=-1).mean().item() - self.input_norm)
        self.output_norm += coeff * (torch.norm(norm_output, dim=-1).mean().item() - self.output_norm)
        self.tokens_processed += num_tokens

        if self.replace_after_steps > 0:
            if self.steps_since_replacement >= self.replace_after_steps:
                self.steps_since_replacement = 0
                self.replace_dead_codes(normalized_input)
            self.steps_since_replacement += 1

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

    def disable_codes(self, codes: Union[Sequence[int], int]):
        """Disable the given codes.

        Args:
            codes: list of codes to disable.
        """
        if isinstance(codes, int):
            codes = [codes]
        self.hook_kwargs["disable_codes"] += codes
        # ensure codes are unique
        self.hook_kwargs["disable_codes"] = list(set(self.hook_kwargs["disable_codes"]))

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

    def replace_dead_codes(self, x: torch.Tensor):
        """re-initialize the dead codebook features and returns number of replaced codes."""
        underused_codes = torch.where(self.counts == 0)[0].to(self.codebook.weight.device)
        num_inactive = len(underused_codes)
        with torch.no_grad():
            x = x.flatten(0, -2)  # flatten to 2D
            x = x[torch.randperm(x.size(0))]  # shuffle
            mult = num_inactive // x.size(0) + 1
            if mult > 1:  # if there's not enough
                x = torch.cat(mult * [x])
            new_codes = x[:num_inactive]

            if self.replace_rho > 0:
                norm = new_codes.norm(p=2, dim=-1, keepdim=True)
                noise = torch.randn_like(new_codes)
                new_codes = new_codes + self.replace_rho * norm * noise

            self.codebook.weight.data[underused_codes] = new_codes
            self.counts.zero_()
        return len(underused_codes)

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
        self._store_data = False

    def store_data(self):
        """Context manager to initialize codebooks using kmeans."""
        self._store_data = True

    def partial_fit_codebook(self):
        """Update the codebook with the data."""
        assert isinstance(self.codebook, KMeansEmbedding)
        self.codebook.partial_fit()


class GroupCodebookLayer(nn.Module):
    """Module that applies distinct codebooks to chunks of input vectors."""

    def __init__(
        self,
        dim: int,
        num_codes: int,
        key: str,
        num_codebooks: int = 1,
        kmeans_init=False,
        snap: bool = True,
        soft_snap: bool = False,
        snap_fn: Type[BaseSnapFunction] = EuclideanSnapFunction,
        hook_fn: Optional[Callable] = None,
        kmeans_kwargs: Optional[Mapping] = None,
        replace_after_steps: int = 0,
        replace_rho: float = 0.0,
        kcodes: int = 1,
    ):
        """Create the group codebook layer.

        Args:
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            key: key to identify the codebook in hook caches.
            num_codebooks: number of codebooks to use in group. Should divide `dim`. Defaults to 1.
            snap: whether to snap (enable) codebooks. Defaults to True.
            soft_snap: whether to snap the input using softmax. Defaults to False.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            hook_fn: hook function apply to codebook ids.
            kmeans_init: whether to initialize the codebook with k-means of the data. Defaults to False.
            kmeans_kwargs: dictionary of arguments to pass to k-means embedding layer.
            replace_after_steps: number of steps after which to replace dead codes. Defaults to 0.
            replace_rho: magnitude of noise to add to new codes. Defaults to 0.0.
            kcodes: number of codebook features to use for computing the output.
        """
        super().__init__()
        if dim % num_codebooks != 0:
            raise ValueError(
                "dim must be divisible by num_codebooks. Got dim: {}, num_codebooks: {}".format(dim, num_codebooks)
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
                    key=key + f"_gcb{i}",
                    kmeans_init=kmeans_init,
                    snap=snap,
                    soft_snap=soft_snap,
                    snap_fn=snap_fn,
                    hook_fn=hook_fn,
                    kmeans_kwargs={**kmeans_kwargs, "random_state": seed + i},
                    replace_after_steps=replace_after_steps,
                    replace_rho=replace_rho,
                    kcodes=kcodes,
                )
                for i in range(num_codebooks)
            ]
        )
        self.hook_fn = hook_fn

    def normalize_L2(self):
        """Normalize the codebook features."""
        for codebook in self.codebook:
            codebook.normalize_L2()

    def use_faiss(self, device=None):
        """Use faiss for snap function."""
        for codebook in self.codebook:
            codebook.use_faiss(device)

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

    def enable_codebook(self):
        """Enable the use of codebook for all the codebooks."""
        for codebook in self.codebook:
            codebook.enable_codebook()

    def disable_codebook(self):
        """Disable the use of codebook for all the codebooks."""
        for codebook in self.codebook:
            codebook.disable_codebook()

    def replace_codes(self):
        """Replace the dead codebook features."""
        avg_replaced_codes = 0
        for codebook in self.codebook:
            avg_replaced_codes += codebook.replace_codes()
        return avg_replaced_codes / len(self.codebook)

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
        return sum(codebook.reconstruction_mse for codebook in self.codebook) / len(self.codebook)

    @property
    def input_norm(self):
        """Return the input norm of the codebooks."""
        return sum(codebook.input_norm for codebook in self.codebook) / len(self.codebook)

    @property
    def output_norm(self):
        """Return the output norm of the codebooks."""
        return sum(codebook.output_norm for codebook in self.codebook) / len(self.codebook)

    def get_most_used_code(self):
        """Return the most used code. Uses the first codebook by default."""
        return self.codebook[0].get_most_used_code()

    def set_hook_fn(self, hook_fn):
        """Set the hook function."""
        self.hook_fn = hook_fn
        for codebook in self.codebook:
            codebook.set_hook_fn(hook_fn)

    def forward(self, x):
        """Snap activations to elements in the codebook.

        Args:
            x: input tensor of shape: (batch_size, seq_len, dim) or (batch_size, seq_len, num_heads, dim).

        Returns: output with the feature vectors replaced using the group codebook.
        """
        if len(x.shape) == 4:
            assert x.shape[2] == self.num_codebooks, f"{x.shape}; num_cbs: {self.num_codebooks}"
            output = torch.stack([self.codebook[i](x[:, :, i]) for i in range(self.num_codebooks)], dim=2)
            return output
        else:
            assert len(x.shape) == 3
            output = torch.cat(
                [self.codebook[i](chunk) for i, chunk in enumerate(x.chunk(self.num_codebooks, dim=-1))],
                dim=-1,
            )
            return output

    def set_hook_kwargs(self, cb_idx=None, **kwargs):
        """Set the hook kwargs for the codebooks."""
        if cb_idx is not None:
            if isinstance(cb_idx, int):
                cb_idx = [cb_idx]
            for i in cb_idx:
                self.codebook[i].set_hook_kwargs(**kwargs)
            return
        for codebook in self.codebook:
            codebook.set_hook_kwargs(**kwargs)

    def disable_codes(self, codes: Union[Sequence[int], int], cb_idx=None):
        """Disable the given codes.

        Args:
            codes: a code or a list of codes to disable.
            cb_idx: index of the head to disable the codes for.
        """
        if cb_idx is not None:
            if isinstance(cb_idx, int):
                cb_idx = [cb_idx]
            for i in cb_idx:
                self.codebook[i].disable_codes(codes)
            return
        for codebook in self.codebook:
            codebook.disable_codes(codes)

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
        return sum(codebook.avg_norm() for codebook in self.codebook) / self.num_codebooks

    def max_norm(self):
        """Return the average norm of the codebook features."""
        return max(codebook.max_norm() for codebook in self.codebook)

    def most_common_counts(self):
        """Return the counts of the codebook features."""
        # num_codes contains total codes across group codebooks
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
            codebook._store_data = False

    def store_data(self):
        """Context manager to initialize codebooks using kmeans."""
        for codebook in self.codebook:
            codebook.store_data()

    def partial_fit_codebook(self):
        """Apply kmeans fit on codebook using the current data."""
        for codebook in self.codebook:
            codebook.partial_fit_codebook()


class CodebookWrapper(nn.Module, abc.ABC):
    """Abstract class to wraps a nn module by applying codebooks on the output of the layer."""

    def __init__(
        self,
        module_layer: nn.Module,
        codebook_cls: Union[
            Type[CodebookLayer],
            Type[GroupCodebookLayer],
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
            module_layer: module layer to wrap codebook on.
            codebook_cls: codebook class to use. Can be either `CodebookLayer` (default),
                `GroupCodebookLayer` or `GroupedCodebookLayer`.
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            key: key to identify the codebook in hook caches.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            num_codebooks: number of codebooks to use in a group. Should divide `dim`. Defaults to 1.
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

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass of the wrapped module."""
        pass

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
            Type[GroupCodebookLayer],
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
            module_layer: module layer to wrap codebook on.
            codebook_cls: codebook class to use. Can be either `CodebookLayer` (default),
                `GroupCodebookLayer` or `GroupedCodebookLayer`.
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            key: key to identify the codebook in hook caches.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            num_codebooks: number of codebooks to use in a group. Should divide `dim`. Defaults to 1.
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
            Type[GroupCodebookLayer],
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
            module_layer: module layer to wrap codebook on.
            codebook_cls: codebook class to use. Can be either `CodebookLayer` (default),
                `GroupCodebookLayer` or `GroupedCodebookLayer`.
            dim: dimension size of the codebook features.
            num_codes: number of codebook features to have.
            key: key to identify the codebook in hook caches.
            snap_fn: snap function to use.
                Can be either `EuclideanSnapFunction` (default) or `InnerProductSnapFunction`.
            num_codebooks: number of codebooks to use in a group. Should divide `dim`. Defaults to 1.
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
        layer_outputs = self.codebook_layer(layer_outputs)
        return layer_outputs


class CodebookModelConfig(transformers.PretrainedConfig):
    """Configuration class to store the configuration of a `CodebookModel`."""

    model_type = "codebook"

    def __init__(
        self,
        codebook_at: Union[str, Sequence] = "mlp",
        codebook_type: Union[str, Sequence] = "vanilla",
        num_codes: int = 100,
        num_codebooks: Union[int, Sequence] = 1,
        k_codebook: Union[int, Sequence] = 1,
        layers_to_snap: Sequence = (),
        similarity_metric: str = "inner_product",
        loss: str = "aeloss",
        kmeans_init: bool = False,
        kmeans_init_examples: int = 1000,
        kmeans_path: Optional[str] = None,
        kmeans_kwargs: Optional[Mapping] = None,
        codebook_kwargs: Optional[Mapping] = None,
        **kwargs,
    ) -> None:
        """Create the config for the codebook model.

        Args:
            codebook_at: where to apply codebook. Can be either 'mlp' (default) or 'transformer_block'.
            codebook_type: type of codebook to use. Can be either 'vanilla' (default, uses `CodebookLayer`)
                or 'group'.
            num_codes: number of codebook features to have.
            num_codebooks: number of codebooks to use in a group. Should divide `dim`. Defaults to 1.
            k_codebook: number of nearest neighbors in codebook snapping.
            layers_to_snap: Index of transformer layers in the model on which codebook to apply.
                Defaults to []. Can contain negative numbers to index from the last layers.
            similarity_metric: similarity metric to use. Can be either 'euclidean' (default) or 'inner_product'.
            loss: whether to use the loss used in VQVAE paper or the CLM loss.
            kmeans_init: whether to initialize codebook with kmeans.
            kmeans_init_examples: number of examples to use for kmeans initialization.
            kmeans_path: path to load or save the kmeans embeddings.
            kmeans_kwargs: additional keyword arguments to pass to kmeans.
            codebook_kwargs: additional keyword arguments to pass to the codebook layer.
            kwargs: additional keyword arguments to pass to the config.
        """
        super().__init__(**kwargs)

        self.codebook_at = codebook_at
        self.codebook_type = codebook_type
        self.num_codes = num_codes
        self.num_codebooks = num_codebooks
        self.k_codebook = k_codebook

        self.layers_to_snap = layers_to_snap
        self.similarity_metric = similarity_metric
        self.loss = loss
        self.kmeans_init = kmeans_init
        self.kmeans_path = kmeans_path
        self.kmeans_init_examples = kmeans_init_examples
        self.kmeans_kwargs = kmeans_kwargs
        self.codebook_kwargs = codebook_kwargs

        self.handle_multiple_codebooks_per_layer()
        self.check_correctness_of_args()

    def handle_multiple_codebooks_per_layer(self):
        """Handle the case when multiple codebooks are applied within layers."""
        if isinstance(self.codebook_at, str):
            self.codebook_at = [self.codebook_at]
        per_layer_codebooks = len(self.codebook_at)

        for key in ["codebook_type", "num_codebooks", "k_codebook", "num_codes"]:
            if isinstance(getattr(self, key), Union[list, tuple]):
                if len(getattr(self, key)) != per_layer_codebooks:
                    raise ValueError(f"length of {key} must match length of `codebook_at`.")
            else:
                setattr(self, key, [getattr(self, key)] * per_layer_codebooks)

    def check_correctness_of_args(self):
        """Ensure that the arguments are correctly specified."""
        for i in range(len(self.codebook_type)):
            if self.codebook_type[i] not in [
                "vanilla",
                "group",
            ]:
                raise ValueError(f"Invalid codebook type {self.codebook_type[i]}")
            if self.codebook_type[i] == "vanilla" and self.num_codebooks[i] != 1:  # type: ignore
                raise ValueError("Vanilla codebook type can only have 1 codebook.")

        if self.loss.split("-")[0] not in ["base", "aeloss", "fullaeloss", "vqvae"]:
            raise ValueError(f"Invalid loss {self.loss}")
        if self.similarity_metric not in ["euclidean", "inner_product"]:
            raise ValueError(f"Invalid similarity metric {self.similarity_metric}")
        if self.codebook_kwargs is None:
            self.codebook_kwargs = {}
        self.replace_codes = False
        if self.codebook_kwargs.get("replace_after_steps", 0) > 0:
            self.replace_codes = True


class CodebookModel(transformers.PreTrainedModel, abc.ABC):
    """ABC for a model containing codebook features.

    Logging metrics is disabled by default for maximum performance. To enable logging,
    use the `enable_logging` method after loading the codebook model.
    """

    config_class = CodebookModelConfig

    def __init__(
        self,
        config: CodebookModelConfig,
        model: nn.Module,
    ) -> None:
        """Build the codebook based model.

        Args:
            config: config for the model.
            model: torch model to apply codebooks to.
        """
        super().__init__(config=config)
        self.codebook_cls: Sequence = []
        self.model: Any = model
        self.logging = True
        self.model_params = list(model.parameters())
        self.codebook_params: Sequence = []
        self.all_codebooks: Mapping[int, Mapping[str, CodebookLayer]] = {}
        self.init_codebook_classes()
        self.set_codebook_args()
        self.add_codebooks()
        # override the forward method
        self.forward = self.model.forward  # type: ignore
        # disable logging by default
        if not self.config.replace_codes:
            self.disable_logging()

    def __setattr__(self, name: str, value: Any) -> None:
        """If the model is set, override the forward method using the new model.

        Required when loading a codebook model using `from_pretrained` method.
        """
        super().__setattr__(name, value)
        if name == "model":
            self.forward = self.model.forward  # type: ignore

    # labels is needed in the signature so that transformers.trainer can return loss
    def forward(self, *args, labels: Optional[torch.LongTensor] = None, **kwargs):
        """Raise an error if this method is called."""
        raise RuntimeError("This shouldn't get executed as forward is overridden in init.")

    def use_faiss(self, device=None):
        """Use FAISS to more efficiently find the top_k codes in each codebook.

        https://github.com/facebookresearch/faiss
        """
        assert faiss is not None, "faiss is not installed."
        for codebooks_dict in self.all_codebooks.values():
            for codebook in codebooks_dict.values():
                codebook.use_faiss(device)

    def init_codebook_classes(self):
        """Initialize the codebook classes based on the `codebook_type` configuration."""
        for cb_type in self.config.codebook_type:
            if cb_type == "vanilla":
                self.codebook_cls.append(CodebookLayer)
            elif cb_type == "group":
                self.codebook_cls.append(GroupCodebookLayer)

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
            raise ValueError("`similarity_metric` should be either 'euclidean' or 'inner_product'.")

    def __getattr__(self, name):
        """Get attributes from the wrapped model if not found in the codebook model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def generate(self, *args, **kwargs):
        """Generate output from the wrapped model."""
        return self.model.generate(*args, **kwargs)

    def get_codebook(self, info: utils.CodeInfo):
        """Get the codebook for the given CodeInfo object."""
        assert info.cb_at is not None
        codebook = self.all_codebooks[info.layer][info.cb_at]
        if info.head is not None:  # grouped codebook
            codebook = codebook.codebook[info.head]
        return codebook

    def add_codebooks(self):
        """Add codebooks for the layers that are to be snapped."""
        CODEBOOK_METHODS = {
            "transformer_block": "codebook_at_transformer",
            "mlp": "codebook_at_mlp",
            "mlp_mid": "codebook_at_mlp_mid",
            "qkv": "codebook_at_qkv",
            "attn": "codebook_at_attention",
            "attn_preproj": "codebook_at_preprojection_attn",
            "attn_plus_mlp": "codebook_at_attention_plus_mlp",
        }
        layers = self.layers()
        for i in range(len(layers)):
            if i in self.config.layers_to_snap:
                codebooks_in_layer = {}
                for i_cb, cb_at in enumerate(self.config.codebook_at):
                    codebook_method = CODEBOOK_METHODS.get(cb_at)
                    if codebook_method is not None:
                        method = getattr(self, codebook_method)
                        codebooks_in_layer[cb_at] = method(layers, i, i_cb)

                if not codebooks_in_layer:
                    raise ValueError(f"Invalid value for `codebook_at`: {self.config.codebook_at}.")
                self.all_codebooks[i] = codebooks_in_layer

    def codebook_at_transformer(self, layers, i, i_cb):
        """Add codebook at the transformer block."""
        layers[i] = TransformerLayerWrapper(
            layers[i],
            codebook_cls=self.codebook_cls[i_cb],
            dim=self.d_model,
            num_codes=self.config.num_codes[i_cb],
            key=f"layer{i}_transformer_block",
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
        return layers[i].codebook_layer

    def codebook_at_mlp(self, layers, i, i_cb):
        """Add codebook at output of the MLP layer."""
        wrapped_mlp = MLPWrapper(
            layers[i].__getattr__(self.mlp_key),
            codebook_cls=self.codebook_cls[i_cb],
            dim=self.d_model,
            num_codes=self.config.num_codes[i_cb],
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
        return wrapped_mlp.codebook_layer

    def codebook_at_mlp_mid(self, layers, i, i_cb):
        """Add codebook at the hidden layer of MLP."""
        mlp = layers[i].__getattr__(self.mlp_key)
        wrapped_hidden_layer = MLPWrapper(
            mlp.__getattr__(self.mlp_mid_key),
            codebook_cls=self.codebook_cls[i_cb],
            dim=self.itermediate_size(),
            num_codes=self.config.num_codes[i_cb],
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
        return wrapped_hidden_layer.codebook_layer

    def codebook_at_qkv(self, layers, i, i_cb):
        """Add a separate codebook at each of the q, k, v layers."""
        attn = layers[i].__getattr__(self.attention_key)
        qkv = attn.__getattr__(self.qkv_key)
        wrapped_hidden_layer = MLPWrapper(
            qkv,
            codebook_cls=GroupCodebookLayer,
            dim=3 * self.d_model,
            num_codes=self.config.num_codes[i_cb],
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
        return wrapped_hidden_layer.codebook_layer

    def codebook_at_attention(self, layers, i, i_cb):
        """Add codebook at the output of the attention layer."""
        wrapped_attn = TransformerLayerWrapper(
            layers[i].__getattr__(self.attention_key),
            codebook_cls=self.codebook_cls[i_cb],
            dim=self.d_model,
            num_codes=self.config.num_codes[i_cb],
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
        return wrapped_attn.codebook_layer

    def codebook_at_preprojection_attn(self, layers, i, i_cb):
        """Add codebook at the attention layer before the output is projected to the residual stream."""
        codebook = self.codebook_cls[i_cb](
            dim=self.d_model,
            num_codes=self.config.num_codes[i_cb],
            key=f"layer{i}_attn_preproj",
            snap_fn=self.snap_fn,
            num_codebooks=self.config.num_codebooks[i_cb],
            kmeans_init=self.config.kmeans_init,
            kmeans_kwargs=self.config.kmeans_kwargs,
            kcodes=self.config.k_codebook[i_cb],
            **self.config.codebook_kwargs,
        )
        extra_args = []
        if (
            not isinstance(self.base_model_cfg(), transformer_lens.HookedTransformerConfig)
            and self.base_model_cfg().model_type == "gpt_neo"
        ):
            attn_key0 = self.attention_key.split(".")[0]
            extra_args = [layers[i].__getattr__(attn_key0).attention_type]

        new_block = self.pre_projection_attn_codebook_cls(
            self.base_model_cfg(),
            *extra_args,
            layer_idx=i,
            codebook_layer=codebook,
        )
        self.codebook_params += list(codebook.parameters())
        if "." in self.attention_key:
            attn_key = self.attention_key.split(".")
            new_block.load_state_dict(
                layers[i].__getattr__(attn_key[0]).__getattr__(attn_key[1]).state_dict(),
                strict=False,
            )
            layers[i].__getattr__(attn_key[0]).__setattr__(attn_key[1], new_block)
        else:
            new_block.load_state_dict(layers[i].__getattr__(self.attention_key).state_dict(), strict=False)
            layers[i].__setattr__(self.attention_key, new_block)
        return codebook

    def codebook_at_attention_plus_mlp(self, layers, i, i_cb):
        """Add codebook on the summed output of attention and MLP layers."""
        codebook = self.codebook_cls[i_cb](
            dim=self.d_model,
            num_codes=self.config.num_codes[i_cb],
            key=f"layer{i}_attn_plus_mlp",
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
        self.codebook_params += list(codebook.parameters())
        pre_res_block.load_state_dict(layers[i].state_dict(), strict=False)
        layers[i] = pre_res_block
        return codebook

    def reset_codebook_metrics(self):
        """Reset the metrics stored in the codebooks."""
        for i, codebooks_dict in self.all_codebooks.items():
            assert i in self.config.layers_to_snap
            for codebook in codebooks_dict.values():
                codebook.reset_metrics()

    def enable_codebooks(self):
        """Enable the codebooks for all layers in self.config.layers_to_snap."""
        for i, layers_dict in self.all_codebooks.items():
            assert i in self.config.layers_to_snap
            for layer in layers_dict.values():
                layer.enable_codebook()

    def disable_codebooks(self):
        """Disable the use of codebooks in all the layers."""
        for i, layers_dict in self.all_codebooks.items():
            assert i in self.config.layers_to_snap
            for layer in layers_dict.values():
                layer.disable_codebook()

    def get_codebook_params(self):
        """Get codebook parameters."""
        return self.codebook_params

    def get_model_params(self):
        """Get model's original parameters (not including codebook params)."""
        return self.model_params

    def set_hook_kwargs(self, idx=None, **kwargs):
        """Set the hook kwargs for the codebook layers in `idx`.

        If `idx` is None, sets the hook kwargs for all codebook layers.
        """
        if idx is not None:
            if isinstance(idx, int):
                idx = [idx]
            for i in idx:
                layers = self.all_codebooks[i].values()
                for layer in layers:
                    layer.set_hook_kwargs(**kwargs)
            return
        for _i, layers in self.all_codebooks.items():
            for layer in layers.values():
                layer.set_hook_kwargs(**kwargs)

    def disable_codes(self, codes: Union[Sequence[int], int], idx=None):
        """Disables the codes in the codebook layers in `idx`.

        If `idx` is None, disables the codes in all codebook layers.
        """
        if idx is not None:
            if isinstance(idx, int):
                idx = [idx]
            for i in idx:
                layers = self.all_codebooks[i]
                for layer in layers.values():
                    layer.disable_codes(codes)
            return
        for _i, layers in self.all_codebooks.items():
            for layer in layers.values():
                layer.disable_codes(codes)

    def reset_hook_kwargs(self, idx=None):
        """Reset the hook kwargs for the codebook layers in `idx`."""
        if idx is not None:
            for layer in list(self.all_codebooks.values())[idx].values():
                layer.reset_hook_kwargs()
            return
        for _i, layers in self.all_codebooks.items():
            for layer in layers.values():
                layer.reset_hook_kwargs()

    def set_hook_fn(self, hook_fn: Callable):
        """Set the hook function to be called after every forward pass of every codebook layer."""
        for i, layers in self.all_codebooks.items():
            assert i in self.config.layers_to_snap
            for layer in layers.values():
                layer.set_hook_fn(hook_fn)

    def get_triggered_codes(self):
        """Get the codes triggered in the last forward pass."""
        triggered_codes = []
        for _i, layers in self.all_codebooks.items():
            for layer in layers.values():
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
        """Get input embeddings of the model."""
        return self.model.get_input_embeddings()

    def save_kmeans_embeddings(self, path):
        """Save kmeans embeddings to a file."""
        state_dict = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if "codebook_layer" in k}
        torch.save(state_dict, path)

    def load_kmeans_embeddings(self, path):
        """Load kmeans embeddings from a file."""
        state_dict = torch.load(path)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if "codebook_layer" in k]
        assert len(unexpected) == 0 and len(missing) == 0

    def init_codebook(self, dataloader):
        """Initialize the codebook weights using kmeans."""
        # check if `kmeans_path` exists and load kmeans embeddings
        if self.config.kmeans_path and os.path.exists(self.config.kmeans_path):
            self.load_kmeans_embeddings(self.config.kmeans_path)
            return

        print("Running kmeans initialization for all the codebooks...")

        self.model.eval()
        self.to(torch.device("cuda"))

        # enable loading training data for kmeans initialization
        for codebooks in self.all_codebooks.values():
            for codebook in codebooks.values():
                codebook.store_data()

        # load data and fit kmeans model
        examples = 0
        with tqdm(total=self.config.kmeans_init_examples) as pbar:
            for data in dataloader:
                if examples >= self.config.kmeans_init_examples:
                    break
                examples += data["input_ids"].shape[0]
                data = {k: v.to(self.device) for k, v in data.items()}
                self.model(**data)
                self.partial_fit_codebook()
                pbar.update(data["input_ids"].shape[0])

        # disable loading data and initialize codebook weights
        for codebooks in self.all_codebooks.values():
            for codebook in codebooks.values():
                codebook.initialize()

        if self.config.kmeans_path and not os.path.exists(self.config.kmeans_path):
            self.save_kmeans_embeddings(self.config.kmeans_path)

    def partial_fit_codebook(self):
        """Fit the codebook to the data stored in the codebook layer."""
        for codebooks in self.all_codebooks.values():
            for codebook in codebooks.values():
                codebook.partial_fit_codebook()

    def enable_logging(self):
        """Enable logging for all the codebooks."""
        self.logging = True
        for codebooks in self.all_codebooks.values():
            for codebook in codebooks.values():
                codebook.enable_logging()

    def disable_logging(self):
        """Disable logging for all the codebooks."""
        self.logging = False
        for codebooks in self.all_codebooks.values():
            for codebook in codebooks.values():
                codebook.disable_logging()

    def replace_codes(self):
        """Replace the dead codebook features."""
        total_replaced_codes = 0
        for codebooks in self.all_codebooks.values():
            avg_replaced_codes = 0
            for codebook in codebooks.values():
                avg_replaced_codes += codebook.replace_codes()
            avg_replaced_codes /= len(codebooks)
            total_replaced_codes += avg_replaced_codes
        total_replaced_codes /= len(self.all_codebooks)
        return total_replaced_codes

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
        """Return the intermediate size of the model."""
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
        """Get the list of transformer layers of the model."""
        pass

    @abc.abstractmethod
    def num_layers(self) -> int:
        """Get the number of transformer layers in the model."""
        pass

    @abc.abstractmethod
    def base_model_cfg(self):
        """Get the base model config."""
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
            config: config for the model.
            model: GPT2 model to apply codebooks to.

        """
        super().__init__(
            config=config,
            model=model,
        )

    def layers(self):
        """Get the list of transformer layers of the model."""
        return self.model.transformer.h  # type: ignore

    def num_layers(self):
        """Get the number of transformer layers in the model."""
        return self.model.config.n_layer  # type: ignore

    def resize_token_embeddings(self, new_num_tokens):
        """Resizes token embeddings of the model."""
        return self.model.resize_token_embeddings(new_num_tokens)

    def itermediate_size(self):
        """Get the intermediate size of the model."""
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
        """Get the base model config."""
        return self.model.config


class GPTNeoXCodebookModel(CodebookModel):
    """Codebook model for GPTNeoX."""

    def __init__(
        self,
        config,
        model,
    ):
        """Build the codebook based model.

        Args:
            config: config for the model.
            model: GPTNeoX model to apply codebooks to.
        """
        super().__init__(
            config=config,
            model=model,
        )

    def layers(self):
        """Get the list of transformer layers of the model."""
        return self.model.gpt_neox.layers

    def num_layers(self):
        """Get the number of transformer layers in the model."""
        return self.model.config.num_hidden_layers

    def resize_token_embeddings(self, new_num_tokens):
        """Resizes token embeddings of the model."""
        raise NotImplementedError("Not implemented for GPTNeoX.")

    def itermediate_size(self):
        """Get the intermediate size of the model."""
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
        """Get the base model config."""
        return self.model.config


class GPTNeoCodebookModel(CodebookModel):
    """Codebook model for GPTNeo."""

    def __init__(
        self,
        config,
        model,
    ):
        """Build the codebook based model.

        Args:
            config: config for the model.
            model: GPTNeo model to apply codebooks to.
        """
        super().__init__(
            config=config,
            model=model,
        )

    def layers(self):
        """Get the list of transformer layers of the model."""
        return self.model.transformer.h

    def num_layers(self):
        """Get the number of transformer layers in the model."""
        return self.model.config.num_layers

    def resize_token_embeddings(self, new_num_tokens):
        """Resizes token embeddings of the model."""
        raise NotImplementedError("Not implemented for GPTNeo.")

    def itermediate_size(self):
        """Get the intermediate size of the model."""
        return self.model.config.intermediate_size

    @property
    def attention_key(self):
        """Returns the attribute name used for attention in the model."""
        return "attn.attention"

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
        return mod_model_classes.PreProjectionAttentionCodebookGPTNeo

    @property
    def pre_residual_codebook_cls(self):
        """Returns the class to use for codebook before residual."""
        raise NotImplementedError("Not implemented for GPTNeo.")

    @property
    def num_heads(self):
        """Returns the number of heads in the model."""
        return self.model.config.num_heads

    @property
    def d_model(self):
        """Returns the dimension of the model."""
        return self.model.config.hidden_size

    def base_model_cfg(self):
        """Get the base model config."""
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
        """Get the list of transformer layers of the model."""
        return self.model.blocks

    def num_layers(self):
        """Get the number of transformer layers in the model."""
        return self.model.cfg.n_layers

    def resize_token_embeddings(self, new_num_tokens):
        """Resizes token embeddings of the model."""
        raise NotImplementedError("Not implemented for HookedTransformer.")

    def itermediate_size(self):
        """Get the intermediate size of the model."""
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
            raise ValueError(f"pre_residual cls not available for {self.model.cfg.model_name}")

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
        """Get the base model config."""
        return self.model.cfg


def wrap_codebook(model_or_path, config=None, pretrained_path=None):
    """Wrap a model with codebooks."""
    if isinstance(model_or_path, str):
        model = transformers.AutoModelForCausalLM.from_pretrained(model_or_path)
    elif isinstance(model_or_path, transformers.PreTrainedModel):
        model = model_or_path
    else:
        raise ValueError("`model_or_path` should be either a string or a PreTrainedModel.")
    model_type_to_cls = {
        "gpt2": GPT2CodebookModel,
        "gpt_neox": GPTNeoXCodebookModel,
        "gpt_neo": GPTNeoCodebookModel,
    }
    cb_model_cls = model_type_to_cls.get(model.config.model_type, None)
    if cb_model_cls is None:
        raise ValueError(f"Model type {model.config.model_type} not supported with codebooks.")
    if pretrained_path is not None:
        return cb_model_cls.from_pretrained(pretrained_path, model)
    if config is None:
        RuntimeWarning("No config provided. Using default config.")
        config = CodebookModelConfig()
    return cb_model_cls(config=config, model=model)


def convert_to_hooked_model(model_path, orig_cb_model, hooked_kwargs=None):
    """Wrap a hooked tranformer model with codebooks."""
    if hooked_kwargs is None:
        hooked_kwargs = {}
    model = transformer_lens.HookedTransformer.from_pretrained(
        model_path,
        **hooked_kwargs,
    )
    if "device" in hooked_kwargs:
        hooked_kwargs.pop("device")
    state_dict = tl_mods.convert_state_dict(orig_cb_model.model, model.cfg)
    model.load_and_process_state_dict(
        state_dict,
        **hooked_kwargs,
    )
    cb_model = HookedTransformerCodebookModel(orig_cb_model.config, model, orig_cb_model.model.config)
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


def convert_to_hooked_model_for_tokfsm(
    model_path,
    orig_cb_model,
    config,
    hooked_kwargs=None,
):
    """Wrap a hooked tranformer model with codebooks."""
    hooked_config = tl_mods.convert_hf_model_config(model_path, config)
    model = transformer_lens.HookedTransformer(hooked_config)
    if hooked_kwargs is None:
        hooked_kwargs = {}
    if "device" in hooked_kwargs:
        hooked_kwargs.pop("device")
    state_dict = tl_mods.convert_state_dict(orig_cb_model.model, model.cfg)  # type: ignore
    model.load_and_process_state_dict(
        state_dict,
        **hooked_kwargs,
    )
    cb_model = HookedTransformerCodebookModel(orig_cb_model.config, model, orig_cb_model.model.config)
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
