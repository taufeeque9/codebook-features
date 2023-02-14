"""Model related classes."""

import abc
from collections import Counter
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import transformers
from sklearn.cluster import KMeans
from torch import nn


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

    def initialize(self, k: int):
        """Initialize the embeddings after all the data is loaded.

        Args:
        ----
            k: number of cluster centers for K-Means.
        """
        kmeans = KMeans(n_clusters=k)
        kmeans = kmeans.fit(self.data.detach().cpu().numpy())
        self._weight = torch.from_numpy(kmeans.cluster_centers_)
        self.data = None


class BaseSnapFunction(torch.autograd.Function):
    """Autograd Fn to snap input to closest codebook feature.
    This is the base class. It should be subclassed with a forward function."""

    vqvae_loss = False
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
        inputs, codebook, codebook_ids, outputs = ctx.saved_tensors
        if BaseSnapFunction.vqvae_loss:
            raise NotImplementedError("VQVAE backward not implemeted with multicode.")
            grad_outputs / 3
            range_idx = (
                torch.arange(codebook.shape[0])
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            range_idx = range_idx.to(inputs.device)
            idx = codebook_ids.unsqueeze(0) == range_idx
            mean_inputs = torch.stack(
                [inputs[idx[i]].mean(0) for i in range(codebook.shape[0])]
            ).to(inputs.device)
            grad_codebook = 2 * torch.nan_to_num(codebook - mean_inputs)
            # straight through estimator + commitment loss gradient
            grad_inputs = grad_outputs + 2 * (inputs - outputs)
        else:
            grad_codebook = torch.autograd.grad(outputs, codebook, grad_outputs)[0]
            # straight through estimator
            grad_inputs = grad_outputs

        return grad_inputs, grad_codebook


class InnerProductSnapFunction(BaseSnapFunction):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, codebook: torch.Tensor):
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
        codebook_ids = logits.topk(BaseSnapFunction.k, dim=-1)[1]

        # enable gradient so that outputs.grad_fn can be used in backward pass.
        with torch.enable_grad():
            outputs = torch.nn.functional.embedding(codebook_ids, codebook)
            outputs = outputs.sum(dim=-2) / BaseSnapFunction.k
        # ctx.save_for_backward(codebook, outputs)
        ctx.save_for_backward(inputs, codebook, codebook_ids, outputs)
        # detach & clone outputs since the returned tensor's grad_fn will be
        # overridden by SnapFunction.backward and we don't want the above
        # outputs.grad_fn to be overridden.
        return outputs.detach().clone(), codebook_ids


class EuclideanSnapFunction(BaseSnapFunction):
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
        logits = -torch.cdist(inputs, codebook, p=2)  # logits are negative distances
        codebook_ids = logits.topk(BaseSnapFunction.k, dim=-1)[1]
        # enable gradient so that outputs.grad_fn can be used in backward pass.
        with torch.enable_grad():
            outputs = torch.nn.functional.embedding(codebook_ids, codebook)
            outputs = outputs.sum(dim=-2) / BaseSnapFunction.k
        # ctx.save_for_backward(codebook, outputs)
        ctx.save_for_backward(inputs, codebook, codebook_ids, outputs)
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
        kmeans_init=False,
        soft_snap: bool = False,
        snap_fn: BaseSnapFunction = EuclideanSnapFunction,
        hook_fn: Callable = None,
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
            self.codebook = KmeansEmbedding(num_embeddings=num_codes, embedding_dim=dim)
        else:
            self.codebook = nn.Embedding(num_embeddings=num_codes, embedding_dim=dim)
        self._num_codes = num_codes
        self.register_buffer("counts", torch.zeros(num_codes, dtype=torch.long))
        self.soft_snap = soft_snap
        self.snap_fn = snap_fn
        self.hook_fn = hook_fn
        self.key = key
        self.reconstruction_mse = 0
        self.tokens_processed = 0

    @property
    def active_codes(self):
        """Return the number of active codes."""
        return len(self.counts)

    @property
    def num_codes(self):
        """Return the total number of codes."""
        return self._num_codes

    def set_hook_fn(self, hook_fn: Callable):
        """Set the hook function.

        Args:
        ----
            hook_fn: hook function to use.
        """
        self.hook_fn = hook_fn

    def forward(self, x):
        """Snaps activations to elements in the codebook.

        Args:
        ----
            x: input tensor of shape: (batch_size, n_channels, dim).

        Returns: output with the feature vectors replaced using the codebook.
        """
        # [batch_size, n_channels, num_codes]
        assert len(x.shape) == 3
        if not self.soft_snap:
            # Hard choice of a single codebook vector
            output, codebook_ids = self.snap_fn.apply(x, self.codebook.weight)
            # update metrics
            # self.counts.update(codebook_ids.cpu().numpy().flat)
            elems, counts = torch.unique(codebook_ids, sorted=False, return_counts=True)
            self.counts[elems] += counts
            coeff = x.shape[0] * x.shape[1]
            coeff /= self.tokens_processed + x.shape[0] * x.shape[1]

            mse_batch = torch.mean(torch.sum(((x - output) ** 2), dim=-1)).item()
            self.reconstruction_mse += coeff * (mse_batch - self.reconstruction_mse)
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

    def reset_metrics(self):
        """Reset the counts of the codebook features."""
        self.counts.zero_()
        self.reconstruction_mse = 0
        self.tokens_processed = 0

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
        self.codebook = nn.ModuleList(
            [
                CodebookLayer(
                    dim // num_codebooks,
                    num_codes,
                    key=key + f"_ccb{i}",
                    kmeans_init=kmeans_init,
                    soft_snap=soft_snap,
                    snap_fn=snap_fn,
                    hook_fn=hook_fn,
                )
                for i in range(num_codebooks)
            ]
        )
        self.hook_fn = hook_fn

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
        output = torch.cat(
            [
                self.codebook[i](chunk)
                for i, chunk in enumerate(x.chunk(self.num_codebooks, dim=-1))
            ],
            dim=-1,
        )
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
        counts = np.zeros((self.num_codes))
        for codebook in self.codebook:
            counts += codebook.most_common_counts()
        return counts


class CodebookWrapper(nn.Module, abc.ABC):
    """Wraps a nn module by applying codebooks on the output of the layer."""

    def __init__(
        self,
        module_layer: nn.Module,
        dim: int,
        num_codes: int,
        key: str,
        snap_fn: BaseSnapFunction = EuclideanSnapFunction,
        num_codebooks: int = 1,
        hook_fn: Callable = None,
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
        self.codebook_layer = CompositionalCodebookLayer(
            dim,
            num_codes,
            key=key,
            snap_fn=snap_fn,
            num_codebooks=num_codebooks,
            hook_fn=hook_fn,
        )
        self.snap = True

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class TransformerLayerWrapper(CodebookWrapper):
    """Wraps a transformer layer module by applying codebooks on the output of the layer."""

    def __init__(
        self,
        module_layer: nn.Module,
        dim: int,
        num_codes: int,
        key: str,
        snap_fn: BaseSnapFunction = EuclideanSnapFunction,
        num_codebooks: int = 1,
        hook_fn: Callable = None,
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
            dim=dim,
            num_codes=num_codes,
            key=key,
            snap_fn=snap_fn,
            num_codebooks=num_codebooks,
            hook_fn=hook_fn,
        )

    def forward(self, *args, **kwargs):
        """Forward function for the wrapped layer.

        Returns: output using codebook features if `snap` is enabled otherwise
            returns the output of the transformer layer.
        """
        layer_outputs = self.module_layer(*args, **kwargs)
        if self.snap:
            snapped_output = self.codebook_layer(layer_outputs[0])
            layer_outputs = (snapped_output, *layer_outputs[1:])

        return layer_outputs


class MLPWrapper(CodebookWrapper):
    """Wraps a MLP layer module by applying codebooks on the output of the layer."""

    def __init__(
        self,
        module_layer: nn.Module,
        dim: int,
        num_codes: int,
        key: str,
        snap_fn: BaseSnapFunction = EuclideanSnapFunction,
        num_codebooks: int = 1,
        hook_fn: Callable = None,
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
            dim=dim,
            num_codes=num_codes,
            key=key,
            snap_fn=snap_fn,
            num_codebooks=num_codebooks,
            hook_fn=hook_fn,
        )

    def forward(self, *args, **kwargs):
        """Forward function for the wrapped layer.

        Returns: output using codebook features if `snap` is enabled otherwise
            returns the output of the transformer layer.
        """
        layer_outputs = self.module_layer(*args, **kwargs)
        if self.snap:
            layer_outputs = self.codebook_layer(layer_outputs)

        return layer_outputs


class PreResidualCodebookGPT2Block(transformers.models.gpt2.modeling_gpt2.GPT2Block):
    def __init__(self, config, layer_idx=None, codebook_layer=None):
        assert not config.add_cross_attention, "Not implemented"
        super().__init__(config, layer_idx)
        self.codebook_layer = codebook_layer
        self.snap = True

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = (
                outputs + cross_attn_outputs[2:]
            )  # add cross attentions if we output attention weights

        # residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        main_stream = feed_forward_hidden_states + attn_output
        if self.codebook_layer and self.snap:
            main_stream = self.codebook_layer(main_stream)
        hidden_states = residual + main_stream

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class PreResidualCodebookGPTNeoXBlock(
    transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXLayer
):
    def __init__(self, config, layer_idx=None, codebook_layer=None):
        assert not config.add_cross_attention, "Not implemented"
        super().__init__(config)
        self.codebook_layer = codebook_layer
        self.snap = True
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):

        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # output_attn: attn_output, present, (attn_weights)
        attn_output = attention_layer_outputs[0]
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            main_stream = mlp_output + attn_output
            if self.codebook_layer and self.snap:
                main_stream = self.codebook_layer(main_stream)
            hidden_states = main_stream + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            # attn_output = attn_output + hidden_states
            mlp_output = self.mlp(
                self.post_attention_layernorm(attn_output + hidden_states)
            )
            main_stream = mlp_output + attn_output
            if self.codebook_layer and self.snap:
                main_stream = self.codebook_layer(main_stream)
            hidden_states = main_stream + hidden_states

        if use_cache:
            outputs = (
                hidden_states,
            ) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs


class CodebookModelConfig(transformers.PretrainedConfig):
    model_type = "codebook"

    def __init__(
        self,
        num_codes: int = 100,
        num_codebooks: int = 1,
        layers_to_snap: Sequence = (),
        similarity_metric: str = "euclidean",
        codebook_at: str = "mlp",
        vqvae_loss: bool = False,
        k_codebook: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_codes = num_codes
        self.num_codebooks = num_codebooks
        self.layers_to_snap = layers_to_snap
        self.similarity_metric = similarity_metric
        self.codebook_at = codebook_at
        if isinstance(codebook_at, str):
            self.codebook_at = codebook_at.lower().split(",")
        self.vqvae_loss = vqvae_loss
        self.k_codebook = k_codebook


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
            model: torch model to apply codebooks to.
            num_codes: number of codebook features to have.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
            layers_to_snap: Index of transformer layers in the model on which codebook to apply.
                Defaults to []. Can contain negative numbers to index from the last layers.
            similarity_metric: similarity metric to use. Can be either 'euclidean' or 'inner_product'.
            codebook_at: where to apply the codebook. Can be either 'mlp' or 'attention'.
            vqvae_loss: whether to use the loss used in VQVAE paper or the CLM loss.
            k_codebook: number of nearest neighbors in codebook snapping.
        """
        super().__init__(config=config)
        self.model = model
        self.model_params = list(model.parameters())

        if self.config.num_codebooks == -1:
            self.config.num_codebooks = self.model.num_heads
        num_layers = self.num_layers()
        if (
            type(self.config.layers_to_snap) is str
            and self.config.layers_to_snap == "all"
        ):
            self.config.layers_to_snap = list(range(num_layers))
        else:
            self.config.layers_to_snap = list(self.config.layers_to_snap)
            for i in range(len(self.config.layers_to_snap)):
                assert -num_layers <= i and i < num_layers
                if self.config.layers_to_snap[i] < 0:
                    self.config.layers_to_snap[i] += num_layers
        self.config.layers_to_snap = sorted(self.config.layers_to_snap)
        self.codebook_params = []
        self.all_codebooks = {}
        # self.freeze_model_params()
        BaseSnapFunction.vqvae_loss = self.config.vqvae_loss
        BaseSnapFunction.k = self.config.k_codebook
        if self.config.similarity_metric == "euclidean":
            self.snap_fn = EuclideanSnapFunction
        elif self.config.similarity_metric == "inner_product":
            self.snap_fn = InnerProductSnapFunction
        else:
            raise ValueError(
                "`similarity_metric` should be either 'euclidean' or 'inner_product'."
            )

    @property
    def device(self):
        return self.model.device

    def add_codebooks(self):
        """Adds codebooks for the layers that are to be snapped."""
        layers = self.layers()
        for i in range(len(layers)):
            if i in self.config.layers_to_snap:
                codebooks_in_layer = []
                if "transformer_block" in self.config.codebook_at:
                    self.codebook_at_transformer(layers, i, codebooks_in_layer)
                if "mlp" in self.config.codebook_at:
                    self.codebook_at_mlp(layers, i, codebooks_in_layer)
                if "mlp_mid" in self.config.codebook_at:
                    self.codebook_at_mlp_mid(layers, i, codebooks_in_layer)
                if "attention" in self.config.codebook_at:
                    self.codebook_at_attention(layers, i, codebooks_in_layer)
                if "attention_and_mlp" in self.config.codebook_at:
                    self.codebook_at_attention_plus_mlp(layers, i, codebooks_in_layer)

                if len(codebooks_in_layer) == 0:
                    raise ValueError(
                        f"Invalid value for `codebook_at`: {self.config.codebook_at}."
                    )
                self.all_codebooks[i] = codebooks_in_layer

    def codebook_at_transformer(self, layers, i, codebooks_in_layer):
        layers[i] = TransformerLayerWrapper(
            layers[i],
            dim=self.model.config.hidden_size,
            num_codes=self.config.num_codes,
            key=f"layer{i}_tb",
            snap_fn=self.snap_fn,
            num_codebooks=self.config.num_codebooks,
        )
        self.codebook_params += list(
            layers[i].codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(layers[i].codebook_layer)

    def codebook_at_mlp(self, layers, i, codebooks_in_layer):
        wrapped_mlp = MLPWrapper(
            layers[i].__getattr__(self.mlp_key),
            dim=self.model.config.hidden_size,
            num_codes=self.config.num_codes,
            key=f"layer{i}_mlp",
            snap_fn=self.snap_fn,
            num_codebooks=self.config.num_codebooks,
        )
        layers[i].__setattr__(self.mlp_key, wrapped_mlp)
        self.codebook_params += list(
            wrapped_mlp.codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(wrapped_mlp.codebook_layer)

    def codebook_at_mlp_mid(self, layers, i, codebooks_in_layer):
        mlp = layers[i].__getattr__(self.mlp_key)
        wrapped_hidden_layer = MLPWrapper(
            mlp.__getattr__(self.mlp_mid_key),
            dim=self.itermediate_size(),
            num_codes=self.config.num_codes,
            key=f"layer{i}_mlp_mid",
            snap_fn=self.snap_fn,
            num_codebooks=self.config.num_codebooks,
        )
        mlp.__setattr__(self.mlp_mid_key, wrapped_hidden_layer)
        self.codebook_params += list(
            wrapped_hidden_layer.codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(wrapped_hidden_layer.codebook_layer)

    def codebook_at_attention(self, layers, i, codebooks_in_layer):
        wrapped_attn = TransformerLayerWrapper(
            layers[i].__getattr__(self.attention_key),
            dim=self.model.config.hidden_size,
            num_codes=self.config.num_codes,
            key=f"layer{i}_attn",
            snap_fn=self.snap_fn,
            num_codebooks=self.config.num_codebooks,
        )
        layers[i].__setattr__(self.attention_key, wrapped_attn)
        self.codebook_params += list(
            wrapped_attn.codebook_layer.codebook.parameters(),
        )
        codebooks_in_layer.append(wrapped_attn.codebook_layer)

    def codebook_at_attention_plus_mlp(self, layers, i, codebooks_in_layer):
        codebooks_in_layer.append(
            CompositionalCodebookLayer(
                dim=self.model.config.hidden_size,
                num_codes=self.config.num_codes,
                key=f"layer{i}_attn+mlp",
                snap_fn=self.snap_fn,
                num_codebooks=self.config.num_codebooks,
            )
        )
        self.codebook_params += list(
            codebooks_in_layer[-1].parameters(),
        )
        layers[i] = self.pre_residual_codebook_cls(
            self.model.config,
            i,
            codebooks_in_layer[-1],
        )

    def reset_codebook_metrics(self):
        """Resets the metrics stored of the codebooks."""
        for i, layers in self.all_codebooks.items():
            assert i in self.config.layers_to_snap
            for layer in layers:
                layer.reset_metrics()

    def enable_codebooks(self):
        """Enable the use of codebooks in all the layers to snap."""
        for i, layers in self.all_codebooks.items():
            assert i in self.config.layers_to_snap
            for layer in layers:
                layer.snap = True

    def disable_codebooks(self):
        """Disable the use of codebooks in all the layers."""
        for i, layers in self.all_codebooks.items():
            assert i in self.config.layers_to_snap
            for layer in layers:
                layer.snap = False

    def get_codebook_params(self):
        """Gets codebook parameters."""
        return self.codebook_params

    def get_model_params(self):
        """Gets model's original parameters (not including codebook params)."""
        return self.model_params

    def set_hook_fn(self, hook_fn: Callable):
        """Sets the hook function to be called after every forward pass of every codebook layer."""
        for i, layers in self.all_codebooks.items():
            assert i in self.config.layers_to_snap
            for layer in layers:
                layer.set_hook_fn(hook_fn)

    # def freeze_model_params(self):
    #     """Freezes model's actual parameters."""
    #     for param in self.get_model_params():
    #         param.requires_grad = False

    # def unfreeze_model_params(self):
    #     """Unfreezes model's actual parameters."""
    #     for param in self.get_model_params():
    #         param.requires_grad = True

    @abc.abstractmethod
    def itermediate_size(self):
        """Returns the intermediate size of the model."""
        pass

    def get_input_embeddings(self):
        """Gets input embeddings of the model."""
        return self.model.get_input_embeddings()

    @property
    @abc.abstractmethod
    def mlp_mid_key(self):
        """Returns the key of layer in MLP layer where codebook is to be applied."""
        pass

    @abc.abstractmethod
    def layers(self):
        """Returns the list of transformer layers of the model."""
        pass

    @abc.abstractmethod
    def num_layers(self):
        """Returns the number of transformer layers in the model."""
        pass


class BertCodebookModel(CodebookModel):
    """Codebook model for Bert-based models."""

    def __init__(
        self,
        model,
        num_codes,
        num_codebooks: int = 1,
        layers_to_snap=(),
        similarity_metric="euclidean",
        codebook_at: str = "mlp",
        vqvae_loss: bool = False,
        k_codebook: int = 1,
    ):
        """Build the codebook based model.

        Args:
        ----
            model: bert model to apply codebooks to.
            num_codes: number of codebook features to have.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
            layers_to_snap: Index of transformer layers in the model on which codebook to apply.
                Defaults to []. Can contain negative numbers to index from the last layers.
            similarity_metric: similarity metric to use. Can be either 'euclidean' (default) or 'inner_product'.
            codebook_at: where to apply codebook. Can be either 'mlp' (default) or 'transformer_block'.
            vqvae_loss: whether to use the loss used in VQVAE paper or the CLM loss.
            k_codebook: number of nearest neighbors in codebook snapping.
        """
        super().__init__(
            model=model,
            num_codes=num_codes,
            num_codebooks=num_codebooks,
            layers_to_snap=layers_to_snap,
            similarity_metric=similarity_metric,
            codebook_at=codebook_at,
            vqvae_loss=vqvae_loss,
            k_codebook=k_codebook,
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
        return self.model.config.num_hidden_layers


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
            model: GPT2 model to apply codebooks to.
            num_codes: number of codebook features to have.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
            layers_to_snap: Index of transformer layers in the model on which codebook to apply.
                Defaults to []. Can contain negative numbers to index from the last layers.
            similarity_metric: similarity metric to use. Can be either 'euclidean' (default) or 'inner_product'.
            codebook_at: where to apply codebook. Can be either 'mlp' (default) or 'transformer_block'.
            vqvae_loss: whether to use the loss used in VQVAE paper or the CLM loss.
            k_codebook: number of nearest neighbors in codebook snapping.
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
            return 4 * self.model.config.hidden_size

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
        return "act"

    @property
    def pre_residual_codebook_cls(self):
        """Returns the class to use for codebook before residual."""
        return PreResidualCodebookGPT2Block

    @property
    def num_heads(self):
        """Returns the number of heads in the model."""
        return self.model.config.n_head


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
            model: GPT2 model to apply codebooks to.
            num_codes: number of codebook features to have.
            num_codebooks: number of codebooks to use compositionally. Should divide `dim`. Defaults to 1.
            layers_to_snap: Index of transformer layers in the model on which codebook to apply.
                Defaults to []. Can contain negative numbers to index from the last layers.
            similarity_metric: similarity metric to use. Can be either 'euclidean' (default) or 'inner_product'.
            codebook_at: where to apply codebook. Can be either 'mlp' (default) or 'transformer_block'.
            vqvae_loss: whether to use the loss used in VQVAE paper or the CLM loss.
            k_codebook: number of nearest neighbors in codebook snapping.
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
    def mlp_key(self):
        """Returns the attribute name used for mlp in the model."""
        return "mlp"

    @property
    def mlp_mid_key(self):
        """Returns the attribute name used for mlp hidden layer in the model."""
        return "act"

    @property
    def pre_residual_codebook_cls(self):
        """Returns the class to use for codebook before residual."""
        return PreResidualCodebookGPTNeoXBlock

    @property
    def num_heads(self):
        """Returns the number of heads in the model."""
        return self.model.config.num_attention_heads


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
