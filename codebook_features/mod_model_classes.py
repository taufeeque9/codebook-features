"""Model classes where codebook is applied by modifying the forward pass of the model."""

from typing import Optional, Tuple, Union

import einops
import torch
import torch.nn.functional as F
from fancy_einsum import einsum
from transformer_lens import components as tl_components
from transformers.models.gpt2 import modeling_gpt2
from transformers.models.gpt_neo import modeling_gpt_neo
from transformers.models.gpt_neox import modeling_gpt_neox


class PreResidualCodebookGPT2Block(modeling_gpt2.GPT2Block):
    """GPT2Block with codebook applied to the main stream."""

    def __init__(self, config, layer_idx=None, codebook_layer=None):
        """Initialize the transformer block."""
        assert not config.add_cross_attention, "Not implemented"
        super().__init__(config, layer_idx)
        self.codebook_layer = codebook_layer

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
        """Forward pass of the transformer block."""
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
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        # residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        main_stream = feed_forward_hidden_states + attn_output
        if self.codebook_layer:
            main_stream = self.codebook_layer(main_stream)
        hidden_states = residual + main_stream

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class PreResidualCodebookGPTNeoXBlock(modeling_gpt_neox.GPTNeoXLayer):
    """GPTNeoXBlock with codebook applied to the main stream."""

    def __init__(self, config, layer_idx=None, codebook_layer=None):
        """Initialize the transformer block."""
        assert not config.add_cross_attention, "Not implemented"
        super().__init__(config)
        self.codebook_layer = codebook_layer
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
        """Forward pass of the transformer block."""
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
            if self.codebook_layer:
                main_stream = self.codebook_layer(main_stream)
            hidden_states = main_stream + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            # attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output + hidden_states))
            main_stream = mlp_output + attn_output
            if self.codebook_layer:
                main_stream = self.codebook_layer(main_stream)
            hidden_states = main_stream + hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs


class PreProjectionAttentionCodebookGPT2(modeling_gpt2.GPT2Attention):
    """GPT2Attention with codebook applied before projecting to the residual stream."""

    def __init__(self, config, layer_idx=None, codebook_layer=None):
        """Initialize the attention layer."""
        super().__init__(config, is_cross_attention=False)
        self.layer_idx = layer_idx
        self.codebook_layer = codebook_layer

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
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """Forward pass of the attention layer."""
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        if self.codebook_layer is not None:
            attn_output = self.codebook_layer(attn_output)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class PreProjectionAttentionCodebookGPTNeoX(modeling_gpt_neox.GPTNeoXAttention):
    """GPTNeoXAttention with codebook applied before projecting to the residual stream."""

    def __init__(self, config, layer_idx=None, codebook_layer=None):
        """Initialize the attention layer."""
        super().__init__(config)
        self.codebook_layer = codebook_layer
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        """Forward pass of the attention layer."""
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = modeling_gpt_neox.apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)

        if self.codebook_layer is not None:
            attn_output = self.codebook_layer(attn_output)

        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class PreProjectionAttentionCodebookGPTNeo(modeling_gpt_neo.GPTNeoSelfAttention):
    """GPTNeoXAttention with codebook applied before projecting to the residual stream."""

    def __init__(self, config, attention_type, layer_idx=None, codebook_layer=None):
        """Initialize the attention layer."""
        super().__init__(config, attention_type)
        self.codebook_layer = codebook_layer
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        """Forward pass of the attention layer."""
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        if self.codebook_layer is not None:
            attn_output = self.codebook_layer(attn_output)

        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class PreProjectionAttentionCodebookHookedTransformer(tl_components.Attention):
    """Hooked Transformer Attention layer with codebook applied before projecting to the residual stream."""

    def __init__(self, config, layer_idx=None, codebook_layer=None):
        """Initialize the attention layer."""
        super().__init__(config, layer_id=layer_idx)
        self.codebook_layer = codebook_layer
        self.layer_idx = layer_idx

    def forward(
        self,
        query_input,
        key_input,
        value_input,
        past_kv_cache_entry=None,
        additive_attention_mask=None,
        left_attention_mask=None,
    ):
        """
        shortformer_pos_embed is only used if self.cfg.positional_embedding_type == "shortformer", else defaults to None and is irrelevant. See HookedTransformerConfig for more details
        past_kv_cache_entry is an optional entry of past keys and values for this layer, only relevant if generating text. Defaults to None
        additive_attention_mask is an optional mask to add to the attention weights. Defaults to None.
        left_attention_mask is the attention mask for left padded tokens. None when right padding is used. Defaults to None.
        """

        if self.cfg.use_split_qkv_input:
            qkv_einops_string = "batch pos head_index d_model"
        else:
            qkv_einops_string = "batch pos d_model"

        q = self.hook_q(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                query_input,
                self.W_Q,
            )
            + self.b_Q
        )  # [batch, pos, head_index, d_head]
        k = self.hook_k(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                key_input,
                self.W_K,
            )
            + self.b_K
        )  # [batch, pos, head_index, d_head]
        v = self.hook_v(
            einsum(
                f"{qkv_einops_string}, head_index d_model d_head \
                -> batch pos head_index d_head",
                value_input,
                self.W_V,
            )
            + self.b_V
        )  # [batch, pos, head_index, d_head]

        if past_kv_cache_entry is not None:
            # Appends the new keys and values to the cached values, and automatically updates the cache
            kv_cache_pos_offset = past_kv_cache_entry.past_keys.size(1)
            k, v = past_kv_cache_entry.append(k, v)
        else:
            # Not using a cache
            kv_cache_pos_offset = 0

        if self.cfg.positional_embedding_type == "rotary":
            q, k = self.rotary_rotate_qk(q, k, kv_cache_pos_offset)

        if self.cfg.dtype not in [torch.float32, torch.float64]:
            # If using 16 bits, increase the precision to avoid numerical instabilities
            q = q.to(torch.float32)
            k = k.to(torch.float32)

        attn_scores = (
            einsum(
                "batch query_pos head_index d_head, \
                    batch key_pos head_index d_head \
                    -> batch head_index query_pos key_pos",
                q,
                k,
            )
            / self.attn_scale
        )  # [batch, head_index, query_pos, key_pos]
        if self.cfg.attention_dir == "causal":
            # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
            attn_scores = self.apply_causal_mask(
                attn_scores, kv_cache_pos_offset, left_attention_mask
            )  # [batch, head_index, query_pos, key_pos]
        if additive_attention_mask is not None:
            attn_scores += additive_attention_mask

        attn_scores = self.hook_attn_scores(attn_scores)
        pattern = F.softmax(attn_scores, dim=-1)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = self.hook_pattern(pattern)  # [batch, head_index, query_pos, key_pos]
        pattern = pattern.to(self.cfg.dtype)
        z = self.hook_z(
            einsum(
                "batch key_pos head_index d_head, \
                batch head_index query_pos key_pos -> \
                batch query_pos head_index d_head",
                v,
                pattern,
            )
        )  # [batch, pos, head_index, d_head]

        if self.codebook_layer is not None:
            z = self.codebook_layer(z)

        if not self.cfg.use_attn_result:
            out = (
                (
                    einsum(
                        "batch pos head_index d_head, \
                            head_index d_head d_model -> \
                            batch pos d_model",
                        z,
                        self.W_O,
                    )
                )
                + self.b_O
            )  # [batch, pos, d_model]
        else:
            # Explicitly calculate the attention result so it can be accessed by a hook
            # This is off by default because it can easily eat through your GPU memory.
            result = self.hook_result(
                einsum(
                    "batch pos head_index d_head, \
                        head_index d_head d_model -> \
                        batch pos head_index d_model",
                    z,
                    self.W_O,
                )
            )  # [batch, pos, head_index, d_model]
            out = (
                einops.reduce(result, "batch position index model->batch position model", "sum") + self.b_O
            )  # [batch, pos, d_model]
        return out
