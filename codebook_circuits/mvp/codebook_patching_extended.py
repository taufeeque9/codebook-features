from typing import Optional, Union, List, Callable, Literal, Dict, Tuple
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint
from codebook_circuits.mvp.more_tl_mods import get_act_name
from codebook_features.models import HookedTransformerCodebookModel
from einops import repeat
import torch as t
import itertools
from functools import partial
from collections import defaultdict
from tqdm import tqdm

# Original implementation from here: https://github.com/callummcdougall/path_patching/blob/main/path_patching.py
_SeqPos = Optional[Int[Tensor, "batch pos"]]
SeqPos = Optional[Union[int, List[int], Int[Tensor, "batch *pos"]]]
IterSeqPos = Union[SeqPos, Literal["each"]]

def hook_fn_generic_patching(
    activation: Float[Tensor, "..."], hook: HookPoint, cache: ActivationCache
) -> Float[Tensor, "..."]:
    """Patches entire tensor of activations, from corresponding values in cache."""
    activation[:] = cache[hook.name][:]
    return activation


def hook_fn_generic_caching(
    activation: Float[Tensor, "..."], hook: HookPoint, name: str = "activation"
) -> Float[Tensor, "..."]:
    """Stores activations in hook context."""
    hook.ctx[name] = activation
    return activation


def hook_fn_generic_patching_from_context(
    activation: Float[Tensor, "..."],
    hook: HookPoint,
    name: str = "activation",
    add: bool = False,
) -> Float[Tensor, "..."]:
    """Patches activations from hook context, if they are there."""
    if name in hook.ctx:
        if add:
            activation = activation + hook.ctx[name]
        else:
            activation[:] = hook.ctx[name][:]
    return activation


class Node:
    """
    Node Implementation from here: https://github.com/callummcdougall/path_patching/blob/main/path_patching.py
    Adapted to work with codebook model.
    """

    def __init__(
        self,
        node_name: str,
        codebook_layer: Optional[int] = None,
        codebook_head_idx: Optional[int] = None,
        sequence_pos: SeqPos = None,
    ):
        partial_cb_activation_str = ".attn.codebook_layer.codebook."

        if partial_cb_activation_str in node_name:
            self.component_name = "cb_" + node_name.split(".")[-2]
            self.codebook_layer = int(node_name.split(".")[1])
        elif "cb_" in node_name:
            self.component_name = node_name
            if codebook_layer is None:
                raise ValueError(
                    "Codebook layer should be given if name given in format 'cb_[codebook_head_idx]'"
                )
            self.codebook_layer = codebook_layer
        else:
            raise ValueError(
                f"Node name {node_name} should be given as full activation name or as 'cb_[codebook_head_idx]'"
            )

        if codebook_head_idx is not None:
            if not isinstance(codebook_head_idx, int):
                raise TypeError(
                    f"Codebook head idx should be an int, not {type(codebook_head_idx)}"
                )
        self.sequence_pos = sequence_pos

    @property
    def activation_name(self) -> str:
        """Return the activation name."""
        return get_act_name(
            name=self.component_name, layer=self.codebook_layer, layer_type="attn"
        )

    def get_patching_hook_fn(
        self,
        cache: Union[str, ActivationCache],
        batch_indices: Union[slice, Int[Tensor, "batch pos"]],
        seq_pos_indices: Union[slice, Int[Tensor, "batch pos"]],
    ) -> Callable:
        """
        Returns a hook function for doing patching according to this node.
        """

        def hook_fn(
            activations: Float[Tensor, "batch seq_pos code"], hook: HookPoint
        ) -> Float[Tensor, "batch pos code"]:
            idx = [batch_indices, seq_pos_indices] + [
                slice(None) for _ in range(activations.ndim - 2)
            ]
            if isinstance(cache, str):
                new_activations = hook.ctx[cache]
            else:
                new_activations = cache[hook.name]
            activations[idx] = new_activations[idx]
            return activations

        return hook_fn

    def __repr__(self):
        """Return a string representation of the Node."""
        return f"Node({self.activation_name})"

def product_with_args_kwargs(*args, **kwargs):
    """
    Helper function which generates an iterable from args and kwargs.

    For example, running the following:
        product_with_args_kwargs([1, 2], ['a', 'b'], key1=[True, False], key2=["q", "k"])
    gives us a list of the following tuples:
        ((1, 'a'), {'key1', True, 'key2': 'q'})
        ((1, 'a'), {'key1', False})
        ((1, 'b'), ('key1', True))
        ...
        ((2, 'b'), {'key1', False, 'key2': 'k'})
    """
    # Generate the product of args
    args_product = list(itertools.product(*args))

    # Generate the product of kwargs values
    kwargs_product = list(itertools.product(*kwargs.values()))

    # Combine args_product with each dict from kwargs_product
    result = []
    for args_values in args_product:
        for kwargs_values in kwargs_product:
            # Construct dict from keys and current values
            kwargs_dict = dict(zip(kwargs.keys(), kwargs_values))
            # Append the tuple with current args_values and kwargs_dict
            result.append((args_values, kwargs_dict))

    return result


class IterNode:
    """
    IterNode Implementation from here: https://github.com/callummcdougall/path_patching/blob/main/path_patching.py
    Adapted to work with codebook model.
    """

    def __init__(self, node_names: Union[str, List[str]], seq_pos: IterSeqPos = None):
        self.seq_pos = seq_pos
        self.component_names = ([node_names] if isinstance(node_names, str) else node_names)
        self.shape_names = {}
        for node in self.component_names:
            self.shape_names[node] = ["layer"]
            if node.startswith("cb_"):
                self.shape_names[node].append("seq_pos")
                self.shape_names[node].append("codes")
            else:
                raise ValueError(f"Node name {node} not recognized")

    def get_node_dict(
            self,
            cb_model: HookedTransformerCodebookModel,
            tensor: Optional[Float[Tensor, "batch seq_pos codes"]] = None,
    ) -> Dict[str, List[Tuple[Float[Tensor, "batch seq_pos codes"], Node]]]:
        """
        This is how we get the actual list of nodes (i.e. `Node` objects) which we'll be iterating through, as well as the seq len (because
        this isn't stored in every node object).

        It will look like:
            {"z": [
                (seq_pos, Node(...)),
                (seq_pos, Node(...)),
                ...
            ],
            "post": [
                (seq_pos, Node(...))
                ...
            ]}
        where each value is a list of `Node` objects, and we'll path patch separately for each one of them.

        We need `model` and `tensor` to do it, because we need to know the shapes of the nodes (e.g. how many layers or sequence positions,
        etc). `tensor` is assumed to have first two dimensions (batch, seq_len).
        """
        batch_size, seq_len = tensor.shape[:2]
        shape_values_all = {
            "seq_pos": seq_len,
            "layer": cb_model.cfg.n_layers,
            "codes": cb_model.config.k_codebook,
        }
        self.nodes_dict = {}
        self.shape_values = {}
        if self.seq_pos == "each":
            seq_pos_indices = list(
                repeat(t.arange(seq_len), "s -> s b 1", b = batch_size)
            )
        else:
            seq_pos_indices = [self.seq_pos]
        
        for node_name, shape_names in self.shape_names.items():
            shape_values_dict = {
                name: value
                for name, value in shape_values_all.items()
                if name in shape_names
            }
            shape_ranges = {
                name: range(value)
                for name, value in shape_values_dict.items()
                if name != "seq_pos"
            }
            shape_values_list = product_with_args_kwargs(seq_pos_indices, **shape_ranges)
            self.nodes_dict[node_name] =[(args[0], Node(node_name, **kwargs)) for args, kwargs in shape_values_list]
        
        return self.nodes_dict

def get_batch_and_seq_pos_indices(seq_pos, batch_size, seq_len):
    """
    seq_pos can be given in four different forms:
        None             -> patch at all positions
        int              -> patch at this position for all sequences in batch
        list / 1D tensor -> this has shape (batch,) and the [i]-th elem is the position to patch for the i-th sequence in the batch
        2D tensor        -> this has shape (batch, pos) and the [i, :]-th elem are all the positions to patch for the i-th sequence

    This function returns batch_indices, seq_pos_indices as either slice(None)'s in the first case, or as 2D tensors in the other cases.

    In other words, if a tensor of activations had shape (batch_size, seq_len, ...), then you could index into it using:

        activations[batch_indices, seq_pos_indices]
    """
    if seq_pos is None:
        seq_sub_pos_len = seq_len
        seq_pos_indices = slice(None)
        batch_indices = slice(None)
    else:
        if isinstance(seq_pos, int):
            seq_pos = [seq_pos for _ in range(batch_size)]
        if isinstance(seq_pos, list):
            seq_pos = t.tensor(seq_pos)
        if seq_pos.ndim == 1:
            seq_pos = seq_pos.unsqueeze(-1)
        assert (
            (seq_pos.ndim == 2)
            and (seq_pos.shape[0] == batch_size)
            and (seq_pos.shape[1] <= seq_len)
            and (seq_pos.max() < seq_len)
        ), "Invalid 'seq_pos' argument."
        seq_sub_pos_len = seq_pos.shape[1]
        seq_pos_indices = seq_pos
        batch_indices = repeat(
            t.arange(batch_size),
            "batch -> batch seq_sub_pos",
            seq_sub_pos=seq_sub_pos_len,
        )

    return batch_indices, seq_pos_indices

def _path_patch_single(
        codebook_model: HookedTransformerCodebookModel,
        orig_input: Union[str, List[str], Int[Tensor, "batch pos"]],
        sender_codebooks: Union[Node, List[Node]],
        receiver_codebooks: Union[Node, List[Node]],
        orig_cache: ActivationCache,
        new_cache: ActivationCache,
        seq_pos: _SeqPos = None) -> Float[Tensor, "batch pos d_vocab"]:
    
    # Clear hooks and context
    codebook_model.reset_hooks()

    # Turn the nodes into list of nodes
    sender_nodes = [sender_codebooks] if isinstance(sender_codebooks, Node) else sender_codebooks
    receiver_nodes = [receiver_codebooks] if isinstance(receiver_codebooks, Node) else receiver_codebooks
    assert isinstance(sender_nodes, list) and isinstance(receiver_nodes, list), "Sender and receiver codebooks should be given as a list of Node objects."

    # Get slices for sequence position
    batch_size, seq_len = orig_cache[get_act_name("cb_0", 1, "attn")].shape[:2]
    batch_indices, seq_pos_indices = get_batch_and_seq_pos_indices(seq_pos, batch_size, seq_len)

    # Run model on orig with sender nodes patched from new and all other nodes frozen. Cache the receiver nodes.

    hooks_for_freezing = []
    hooks_for_caching_receivers = []
    hooks_for_patching_senders = []

    # Get all the hooks we need for freezing codebooks 
    hooks_for_freezing.append(
        (
        lambda name: name.endswith(".hook_codebook_ids"),
         partial(hook_fn_generic_patching, cache = orig_cache),
         )
    )

    print(hooks_for_freezing)

    # Get all the hooks we need for patching senders
    for node in sender_nodes:
        hooks_for_patching_senders.append(
            (
            node.activation_name,
            node.get_patching_hook_fn(
            new_cache, batch_indices, seq_pos_indices
            ),
            )
        )

    print(hooks_for_patching_senders)
    # Get all the hooks we need for caching receiver nodes
    for node in receiver_nodes:
        hooks_for_caching_receivers.append(
            (
                node.activation_name,
                partial(hook_fn_generic_caching, name="receiver_activations"),
            )
        )
    print(hooks_for_caching_receivers)
    # Now add all the hooks in order. Note that patching should override freezing, and caching should happen before both.
    codebook_model.run_with_hooks(
        orig_input,
        return_type=None,
        fwd_hooks=hooks_for_caching_receivers
        + hooks_for_freezing
        + hooks_for_patching_senders,
        clear_contexts=False,  # This is the default anyway, but just want to be sure!
    )
    # Result - we've now cached the receiver nodes (i.e. stored them in the appropriate hook contexts)

    # Lastly, we add the hooks for patching receivers (this is a bit different depending on our alg)
    for node in receiver_nodes:
        codebook_model.add_hook(
            node.activation_name,
            node.get_patching_hook_fn(
                "receiver_activations", batch_indices, seq_pos_indices
            ),
            level=1,
        )
    
    patched_logits = codebook_model(orig_input)
    return patched_logits

def path_patch(
    codebook_model: HookedTransformerCodebookModel,
    orig_input: Union[str, List[str], Int[Tensor, "batch pos"]],
    new_input: Optional[Union[str, List[str], Int[Tensor, "batch pos"]]] = None,
    sender_nodes: Union[Node, List[Node]] = [],
    receiver_nodes: Union[Node, List[Node]] = [],
    orig_cache: ActivationCache = None,
    new_cache: ActivationCache = None,
    seq_pos: _SeqPos = None,
    verbose: bool = False) ->  Float[Tensor, "batch pos d_vocab"]:
    """
    Performs a single instance / multiple instances of path patching, from sender node(s) to receiver node(s).

    Note, I'm using orig and new in place of clean and corrupted to avoid ambiguity. In the case of noising algs (which patching usually is),
    orig=clean and new=corrupted.


    Args:
        model:
            The model we patch with

        orig_input:
            The original input to the model (string, list of strings, or tensor of tokens)

        new_input:
            The new input to the model (string, list of strings, or tensor of tokens)
            i.e. we're measuring the effect of changing the given path from orig->new

        sender_nodes:
            The nodes in the path that come first (i.e. we patch the path from sender to receiver).
            This is given as a `Node` instance, or list of `Node` instances. See the `Node` class for more details.
            Note, if it's a list, they are all treated as senders (i.e. a single value is returned), rather than one-by-one.

        receiver_nodes:
            The nodes in the path that come last (i.e. we patch the path from sender to receiver).
            This is given as a `Node` instance, or list of `Node` instances. See the `Node` class for more details.
            Note, if it's a list, they are all treated as receivers (i.e. a single value is returned), rather than one-by-one.

        patching_metric:
            Should take in a tensor of logits, and output a scalar tensor.
            This is how we calculate the value we'll return.

        apply_metric_to_cache:
            If True, then we apply the metric to the cache we get on the final patched forward pass, rather than the logits.

        verbose:
            Whether to print out extra info (in particular, about the shape of the final output).

    Returns:
        Scalar tensor (i.e. containing a single value).

    ===============================================================
    How we perform multiple instances of path patching:
    ===============================================================

    We can also do multiple instances of path patching in sequence, i.e. we fix our sender node(s) and iterate over receiver nodes,
    or vice-versa.

    The way we do this is by using a `IterNode` instance, rather than a `Node` instance. For instance, if we want to fix receivers
    and iterate over senders, we would use a `IterNode` instance for receivers, and a `Node` instance for senders.

    See the `IterNode` class for more info on how we can specify multiple nodes.

    Returns:
        Dictionary of tensors, keys are the node names of whatever we're iterating over.
        For instance, if we're fixing a single sender head, and iterating over (mlp_out, attn_out) for receivers, then we'd return a dict
        with keys "mlp_out" and "attn_out", and values are the tensors of patching metrics for each of these two receiver nodes.
    """
    # Make sure we aren't iterating over both senders and receivers
    assert not all(
        [isinstance(sender_nodes, IterNode), isinstance(receiver_nodes, IterNode)]
    ), "Can't iterate over both senders and receivers!"

    # Check other arguments
    assert any(
        [isinstance(new_cache, ActivationCache), new_cache == "zero", new_cache is None]
    ), "Invalid new_cache argument."
    assert sender_nodes != [], "You must specify sender nodes."
    assert receiver_nodes != [], "You must specify receiver nodes."

    # ========== Step 1 ==========
    # Gather activations on orig and new distributions (we only need attn heads and possibly MLPs)
    # This is so that we can patch/freeze during step 2
    codebook_name_filter = lambda name: name.endswith(f".hook_codebook_ids")

    if orig_cache is None:
        _, orig_cache = codebook_model.run_with_cache(orig_input, return_type=None)
    if new_cache == "zero":
        new_cache = ActivationCache(
            {k: t.zeros_like(v) for k, v in orig_cache.items()}, model=codebook_model
        )
    elif new_cache is None:
        _, new_cache = codebook_model.run_with_cache(
            new_input, return_type=None, names_filter=codebook_name_filter
        )
    
    # Get out backend patching function (fix all the arguments we won't be changing)
    path_patch_single = partial(
        _path_patch_single,
        codebook_model=codebook_model,
        orig_input=orig_input,
        # sender_codebooks=sender_codebooks,
        # receiver_codebooks=receiver_codebooks,
        orig_cache=orig_cache,
        new_cache=new_cache,
        # seq_pos=seq_pos,
    )

    # Case where we don't iterate, just single instance of path patching:
    if not any(
        [isinstance(receiver_nodes, IterNode), isinstance(sender_nodes, IterNode)]
    ):
        return path_patch_single(
            sender_codebooks=sender_nodes, receiver_codebooks=receiver_nodes, seq_pos=seq_pos
        )
    
    # Case where we're iterating: either over senders, or over receivers
    assert (
        seq_pos is None
    ), "Can't specify seq_pos if you're iterating over nodes. Should use seq_pos='all' or 'each' in the IterNode class."
    results_dict = defaultdict()

    # If we're fixing sender(s), and iterating over receivers:
    if isinstance(receiver_nodes, IterNode):
        receiver_nodes_dict = receiver_nodes.get_node_dict(codebook_model, new_cache[get_act_name("cb_0", 0, "attn")])
        progress_bar = tqdm(
            total=sum(len(node_list) for node_list in receiver_nodes_dict.values())
        )
        for receiver_node_name, receiver_node_list in receiver_nodes_dict.items():
            progress_bar.set_description(f"Patching over {receiver_node_name!r}")
            results_dict[receiver_node_name] = []
            for seq_pos, receiver_node in receiver_node_list:
                results_dict[receiver_node_name].append(
                    path_patch_single(
                        sender_codebooks=sender_nodes, receiver_codebooks=receiver_node, seq_pos=seq_pos
                    )
                )
                progress_bar.update(1)
        progress_bar.close()
        for node_name, node_shape_dict in receiver_nodes.shape_values.items():
            if verbose:
                print(
                    f"results[{node_name!r}].shape = ({', '.join(f'{s}={v}' for s, v in node_shape_dict.items())})"
                )
        return {
            node_name: t.tensor(results).reshape(
                list(receiver_nodes.shape_values[node_name].values())
            )
            if isinstance(results[0], float)
            else results
            for node_name, results in results_dict.items()
        }

    # If we're fixing receiver(s), and iterating over senders:
    elif isinstance(sender_nodes, IterNode):
        sender_nodes_dict = sender_nodes.get_node_dict(codebook_model, new_cache[get_act_name("cb_0", 0, "attn")])
        progress_bar = tqdm(
            total=sum(len(node_list) for node_list in sender_nodes_dict.values())
        )
        for sender_node_name, sender_node_list in sender_nodes_dict.items():
            progress_bar.set_description(f"Patching over {sender_node_name!r}")
            results_dict[sender_node_name] = []
            for seq_pos, sender_node in sender_node_list:
                results_dict[sender_node_name].append(
                    path_patch_single(
                        sender_codebooks=sender_node, receiver_codebooks=receiver_nodes, seq_pos=seq_pos
                    )
                )
                progress_bar.update(1)
                t.cuda.empty_cache()
        progress_bar.close()
        for node_name, node_shape_dict in sender_nodes.shape_values.items():
            if verbose:
                print(
                    f"results[{node_name!r}].shape = ({', '.join(f'{s}={v}' for s, v in node_shape_dict.items())})"
                )
        return {
            node_name: t.tensor(results).reshape(
                list(sender_nodes.shape_values[node_name].values())
            )
            if isinstance(results[0], float)
            else results
            for node_name, results in results_dict.items()
        }








    
