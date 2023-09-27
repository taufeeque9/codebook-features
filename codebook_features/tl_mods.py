"""Modifications to functions from the transformer_lens library."""

import transformer_lens
from transformer_lens.loading_from_pretrained import get_official_model_name
from transformers import AutoConfig

# Taken from the commit:
# https://github.com/neelnanda-io/TransformerLens/commit/54d548de4995a1ecc5b01b9c03aceaf0966c0eb3
# The function `convert_state_dict` is added for convenience.
# The function `convert_hf_model_config` is modified to support unofficial model names provided that
# the `hf_config` is provided.


def convert_state_dict(model, cfg: transformer_lens.HookedTransformerConfig):
    """Convert a state_dict from a HuggingFace model to a state_dict compatible with HookedTransformer."""
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


def convert_hf_model_config(model_name: str, hf_config=None, **kwargs):
    """Return the model config for a HuggingFace model, as a dict in the HookedTransformerConfig format."""
    # In case the user passed in an alias
    try:
        official_model_name = get_official_model_name(model_name)
    except ValueError:
        official_model_name = model_name
    # Load HuggingFace model config
    if "llama" not in official_model_name.lower():
        if hf_config is None:
            hf_config = AutoConfig.from_pretrained(official_model_name, **kwargs)
        architecture = hf_config.architectures[0]
    else:
        architecture = "LlamaForCausalLM"
    if official_model_name.startswith(
        ("llama-7b", "Llama-2-7b")
    ):  # same architecture for LLaMA and Llama-2
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 2048 if official_model_name.startswith("llama-7b") else 4096,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif official_model_name.startswith(
        ("llama-13b", "Llama-2-13b")
    ):  # same architecture for LLaMA and Llama-2
        cfg_dict = {
            "d_model": 5120,
            "d_head": 5120 // 40,
            "n_heads": 40,
            "d_mlp": 13824,
            "n_layers": 40,
            "n_ctx": 2048 if official_model_name.startswith("llama-13b") else 4096,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 5120 // 40,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-30b" in official_model_name:
        cfg_dict = {
            "d_model": 6656,
            "d_head": 6656 // 52,
            "n_heads": 52,
            "d_mlp": 17920,
            "n_layers": 60,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 6656 // 52,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-65b" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 8192 // 64,
            "n_heads": 64,
            "d_mlp": 22016,
            "n_layers": 80,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 8192 // 64,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "GPTNeoForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_heads,
            "n_heads": hf_config.num_heads,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.num_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "attn_types": hf_config.attention_layers,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": False,
            "use_local_attn": True,
            "window_size": hf_config.window_size,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPT2LMHeadModel":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_ctx,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
        }
    elif architecture == "OPTForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.ffn_dim,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPTJForCausalLM":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": 4 * hf_config.n_embd,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_positions,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.rotary_dim,
            "normalization_type": "LN",
        }
    elif architecture == "GPTNeoXForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "normalization_type": "LN",
        }
        rotary_pct = hf_config.rotary_pct
        cfg_dict["rotary_dim"] = round(rotary_pct * cfg_dict["d_head"])
    elif architecture == "BertForMaskedLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": "gelu",
            "attention_dir": "bidirectional",
        }
    else:
        raise NotImplementedError(f"{architecture} is not currently supported.")
    # All of these models use LayerNorm
    cfg_dict["original_architecture"] = architecture
    # The name such that AutoTokenizer.from_pretrained works
    cfg_dict["tokenizer_name"] = official_model_name
    return cfg_dict
