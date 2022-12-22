"""Model related classes."""

import abc
from collections import Counter

import torch
from sklearn.cluster import KMeans
from torch import nn


# A version of the embedding module that can be initialized
# to the kmeans of batched data
class KmeansEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None,
    ):
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

    # Load a batch of data
    def load_data(self, data):
        with torch.no_grad():
            if self.data is None:
                self.data = data
            else:
                self.data = torch.cat([self.data, data], dim=0)

    # After the data has been loaded, call initialize
    def initialize(self, k):
        kmeans = KMeans(n_clusters=k)
        kmeans = kmeans.fit(self.data.detach().cpu().numpy())
        self._weight = torch.from_numpy(kmeans.cluster_centers_)
        self.data = None


class SnapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, codebook):
        logits = torch.matmul(input, codebook.T)
        codebook_ids = logits.argmax(-1)
        with torch.enable_grad():
            output = torch.nn.functional.embedding(codebook_ids, codebook)
        ctx.save_for_backward(codebook, output)
        return output.detach().clone(), codebook_ids

    @staticmethod
    def backward(ctx, grad_output, grad_codebook_ids):
        codebook, output = ctx.saved_tensors
        grad_codebook = torch.autograd.grad(output, codebook, grad_output)[0]
        # straight through estimator
        return grad_output, grad_codebook


class CodebookLayer(nn.Module):
    def __init__(self, dim, num_codes, kmeans_init=False, tau=0):
        super().__init__()
        if kmeans_init:
            self.codebook = KmeansEmbedding(num_embeddings=num_codes, embedding_dim=dim)
        else:
            self.codebook = nn.Embedding(num_embeddings=num_codes, embedding_dim=dim)
        self.counts = Counter()
        self.tau = tau

    def get_code(self, x):
        # [batch_size, n_channels, num_codes]
        logits = torch.matmul(x, self.codebook.weight.T)
        # Was previously doing gumbel softmax--not anymore.
        # codebook_ids = torch.nn.functional.gumbel_softmax(
        #   logits, hard=True).argmax(-1)
        codebook_ids = logits.argmax(-1)
        return codebook_ids

    def forward(self, x):
        """Snaps activations to elements in the codebook.
        Expects input x of size [batch_size, n_channels, dim].
        """
        # [batch_size, n_channels, num_codes]
        assert len(x.shape) == 3
        if self.tau <= 0:
            # Hard choice of a single codebook vector
            output, codebook_ids = SnapFunction.apply(x, self.codebook.weight)
            self.counts.update(codebook_ids.cpu().numpy().flat)
        elif self.tau == 1:
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

    def expire_code(self, code_id):
        # Generate a new code
        # replace code_id
        pass

    # TODO: Consider using a fraction for the threshold instead of an absolute number
    def expire_codes(self, threshold=1):
        underused_codes = set()
        for i in range(self.codebook.weight.size(0)):
            if i not in self.counts or self.counts[i] < threshold:
                underused_codes.add(i)
        with torch.no_grad():
            weights = torch.rand((len(underused_codes), self.codebook.weight.size(0)))
            weights = weights / weights.sum(1, keepdim=True)
            weights = weights.to(self.codebook.weight.device)
            new_codes = torch.einsum("uc,cd->ud", weights, self.codebook.weight)
            underused_codes = torch.tensor(list(underused_codes)).to(
                self.codebook.weight.device
            )
            try:
                self.codebook.weight[underused_codes] = new_codes
            except IndexError:
                pass


class TransformerLayerWrapper(nn.Module):
    def __init__(self, transformer_layer, snap=False, dim=None, num_codes=None):
        super().__init__()
        self.transformer_layer = transformer_layer
        self.snap = snap
        self.codebook_layer = CodebookLayer(dim, num_codes)

    def forward(self, *args, **kwargs):
        layer_outputs = self.transformer_layer(*args, **kwargs)
        if self.snap:
            snapped_output = self.codebook_layer(layer_outputs[0])
            layer_outputs = (snapped_output, *layer_outputs[1:])

        return layer_outputs


class CodebookModel(nn.Module, abc.ABC):
    def __init__(self, model, num_codes, snap_layers=[]) -> None:
        super().__init__()
        self.model = model
        self.num_codes = num_codes
        self.snap_layers = snap_layers
        num_layers = self.num_layers()
        for i in range(len(self.snap_layers)):
            assert -num_layers <= i and i < num_layers
            if self.snap_layers[i] < 0:
                self.snap_layers[i] += self.num_layers()
        self.snap_layers = sorted(self.snap_layers)
        self.codebook_params = []

    def add_codebooks(self):
        layers = self.layers()
        for i in range(len(layers)):
            if i in self.snap_layers:
                layers[i] = TransformerLayerWrapper(
                    layers[i],
                    snap=True,
                    dim=self.model.config.hidden_size,
                    num_codes=self.num_codes,
                )
                self.codebook_params += list(
                    layers[i].codebook_layer.codebook.parameters()
                )

    def get_codebook_params(self):
        return self.codebook_params

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    @abc.abstractmethod
    def layers(self):
        pass

    @abc.abstractmethod
    def num_layers(self):
        pass


class BERTCodebookModel(CodebookModel):
    def __init__(self, model, num_codes, snap_layers=[]):
        super().__init__(model, num_codes, snap_layers)
        self.add_codebooks()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def num_layers(self):
        return self.model.config.num_hidden_layers

    def layers(self):
        return self.model.bert.encoder.layer


class GPT2CodebookModel(CodebookModel):
    def __init__(self, model, num_codes, snap_layers=[]):
        super().__init__(model, num_codes, snap_layers)
        self.add_codebooks()
        self.forward = self.model.forward

    # def forward(self, *args, **kwargs):
    #     return self.model(*args, **kwargs)

    def use_codebooks(self):
        for layer in self.layers():
            layer.snap = True

    def use_orig_model(self):
        for layer in self.layers():
            layer.snap = False

    def layers(self):
        return self.model.transformer.h

    def num_layers(self):
        return self.model.config.n_layer
