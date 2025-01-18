import logging
from functools import partial

import jax
import jax.numpy as jnp
import einops
from typing import Optional
from jax.sharding import PartitionSpec as PS
from jax.experimental.pjit import pjit
from EasyLM.jax_utils import with_sharding_constraint




def flash_attention(
        query: jnp.ndarray,  # [batch, seq_len, num_q_heads, head_dim]
        key: jnp.ndarray,  # [batch, seq_len, num_kv_heads, head_dim]
        value: jnp.ndarray,  # [batch, seq_len, num_kv_heads, head_dim]
        bias: Optional[jnp.ndarray] = None,
        causal: bool = True,
        chunk_size: int = 128,
):
    """JAX implementation of Flash Attention with GQA support and TPU sharding.

    Expected sharding patterns:
    - Input tensors: PS(("dp", "fsdp"), None, "mp", None)
    - Intermediate tensors maintain similar patterns
    - Output: PS(("dp", "fsdp"), None, "mp", None)

    Args:
        query: Query tensor with shape [batch, seq_len, num_q_heads, head_dim]
        key: Key tensor with shape [batch, seq_len, num_kv_heads, head_dim]
        value: Value tensor with shape [batch, seq_len, num_kv_heads, head_dim]
        bias: Optional attention bias
        causal: Whether to apply causal masking
        chunk_size: Size of chunks for blocked computation

    Returns:
        Output tensor with shape [batch, seq_len, num_q_heads, head_dim]
    """
    jax.debug.print("Starting Flash Attention")

    batch_size, seq_len, num_q_heads, head_dim = query.shape
    _, _, num_kv_heads, _ = key.shape

    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"Number of query heads ({num_q_heads}) must be divisible by "
            f"number of key/value heads ({num_kv_heads})"
        )

    num_groups = num_q_heads // num_kv_heads

    # Scale query
    scale = jnp.sqrt(head_dim).astype(query.dtype)
    query = query / scale

    # Split sequence into chunks and maintain sharding
    query = einops.rearrange(query, 'b (n c) h d -> b n c h d', c=chunk_size)
    key = einops.rearrange(key, 'b (n c) h d -> b n c h d', c=chunk_size)
    value = einops.rearrange(value, 'b (n c) h d -> b n c h d', c=chunk_size)

    # Apply sharding constraints after reshape
    query = with_sharding_constraint(query, PS(("dp", "fsdp"), None, None, "mp", None))
    key = with_sharding_constraint(key, PS(("dp", "fsdp"), None, None, "mp", None))
    value = with_sharding_constraint(value, PS(("dp", "fsdp"), None, None, "mp", None))

    # Define kv chunk scanner with pjit
    # @partial(pjit,
    #          in_shardings=(PS(("dp", "fsdp"), "mp", None, None),  # for m_inner
    #                        PS(("dp", "fsdp"), "mp", None, None),  # for l_inner
    #                        PS(("dp", "fsdp"), None, "mp", None),  # for o_inner
    #                        None),  # for idx_k
    #          out_shardings=(PS(("dp", "fsdp"), "mp", None, None),  # for new m
    #                         PS(("dp", "fsdp"), "mp", None, None),  # for new l
    #                         PS(("dp", "fsdp"), None, "mp", None)))  # for new o
    def kv_chunk_scanner(query_chunk, m_inner, l_inner, o_inner, idx_k, idx_n):
        from EasyLM.jax_utils import debug_tensor, create_debug_gather_fn
        
        # Get key/value chunks and repeat for each query group
        k = einops.repeat(
            key[:, idx_k],  # [batch, chunk_size, num_kv_heads, head_dim]
            'b c h d -> b c (h g) d', g=num_groups
        )
        v = einops.repeat(
            value[:, idx_k],  # [batch, chunk_size, num_kv_heads, head_dim]
            'b c h d -> b c (h g) d', g=num_groups
        )

        # Debug prints to track computation
        debug_tensor(f"Processing chunk q={idx_n} k={idx_k}", k)

        # Apply sharding after repeat
        k = with_sharding_constraint(k, PS(("dp", "fsdp"), None, "mp", None))
        v = with_sharding_constraint(v, PS(("dp", "fsdp"), None, "mp", None))

        # Compute attention scores for this block
        # query_chunk shape: [batch, chunk_size, num_heads, head_dim]
        # k shape: [batch, chunk_size, num_heads, head_dim]
        scores = jnp.einsum('bchd,bkhd->bhck', query_chunk, k)
        scores = with_sharding_constraint(scores, PS(("dp", "fsdp"), "mp", None, None))

        from EasyLM.jax_utils import debug_sharded, create_debug_gather_fn
        # Create gather functions with appropriate partition specs
        query_gather_fn = create_debug_gather_fn(partition_spec=PS(("dp", "fsdp"), None, "mp", None))
        scores_gather_fn = create_debug_gather_fn(partition_spec=PS(("dp", "fsdp"), "mp", None, None))

        debug_sharded("Query", query, gather_fn=query_gather_fn)
        debug_sharded("Key", k, gather_fn=query_gather_fn)  # Same spec as query
        debug_sharded("Raw scores", scores, gather_fn=scores_gather_fn)

        if bias is not None:
            # Handle GQA bias if provided
            if bias.shape[1] == num_q_heads:
                bias_block = jax.lax.dynamic_slice(
                    bias,
                    (0, 0, idx_n * chunk_size, idx_k * chunk_size),
                    (bias.shape[0], bias.shape[1], chunk_size, chunk_size)
                )
            elif bias.shape[1] == num_kv_heads:
                bias_block = jax.lax.dynamic_slice(
                    bias,
                    (0, 0, idx_n * chunk_size, idx_k * chunk_size),
                    (bias.shape[0], bias.shape[1], chunk_size, chunk_size)
                )
                bias_block = einops.repeat(
                    bias_block, 'b h q k -> b (h g) q k', g=num_groups
                )
            scores = scores + bias_block

        if causal:
            q_pos = idx_n * chunk_size + jnp.arange(chunk_size)
            k_pos = idx_k * chunk_size + jnp.arange(chunk_size)
            causal_mask = q_pos[:, None] < k_pos[None, :]
            scores = jnp.where(
                causal_mask[None, None, :, :],
                jnp.finfo(scores.dtype).min,
                scores
            )

        m_new = jnp.maximum(m_inner, scores.max(-1, keepdims=True))
        scores = jnp.exp(scores - m_new)
        scores_gather_fn = create_debug_gather_fn(partition_spec=PS(("dp", "fsdp"), "mp", None, None))
        debug_sharded("Post-softmax scores", scores, gather_fn=scores_gather_fn)
        l_new = l_inner * jnp.exp(m_inner - m_new) + scores.sum(-1, keepdims=True)
        # Reshape m_new for proper broadcasting
        scale = jnp.exp(m_inner - m_new)  # [batch, num_heads, chunk_size, 1]
        # Reshape scale to match o_inner dimensions
        scale = jnp.expand_dims(scale, axis=-1)  # Add dim for head_dim
        scale = jnp.transpose(scale, (0, 2, 1, 3, 4))  # Reorder to (batch, num_heads, chunk_size, 1, 1)

        o_new = (o_inner * scale[..., 0] +  # Remove last dim after broadcast
                 jnp.einsum('bhqk,bkhd->bqhd', scores, v))

        # Apply sharding to intermediate results
        o_new = with_sharding_constraint(o_new, PS(("dp", "fsdp"), None, "mp", None))

        return (m_new, l_new, o_new), None

    # Define chunk scanner with pjit
    # @partial(pjit,
    #         in_shardings=(PS(("dp", "fsdp"), "mp", None, None),  # for carry m
    #                      PS(("dp", "fsdp"), "mp", None, None),  # for carry l
    #                      PS(("dp", "fsdp"), None, "mp", None),  # for carry o
    #                      None),  # for idx_n
    #         out_shardings=((PS(("dp", "fsdp"), "mp", None, None),  # for new m
    #                       PS(("dp", "fsdp"), "mp", None, None),  # for new l
    #                       PS(("dp", "fsdp"), None, "mp", None)),  # for new o
    #                      PS(("dp", "fsdp"), None, "mp", None)))  # for output o
    def chunk_scanner(m, l, o, idx_n):
        # Simple debug print with just the index
        jax.debug.print("Chunk scanner idx: {idx}", idx=idx_n)

        # Get current query chunk and maintain sharding
        q = query[:, idx_n]  # [batch, chunk_size, num_q_heads, head_dim]
        q = with_sharding_constraint(q, PS(("dp", "fsdp"), None, "mp", None))

        def kv_chunk_scanner(m_inner, l_inner, o_inner, idx_k, idx_n):
            # Debug prints to verify mechanism works
            jax.debug.print("Processing kv chunk with idx_k={k}", k=idx_k)
            
            # Get key/value chunks and repeat for each query group
            k = einops.repeat(
                key[:, idx_k],  # [batch, chunk_size, num_kv_heads, head_dim]
                'b c h d -> b c (h g) d', g=num_groups
            )
            v = einops.repeat(
                value[:, idx_k],  # [batch, chunk_size, num_kv_heads, head_dim]
                'b c h d -> b c (h g) d', g=num_groups
            )

            # Apply sharding after repeat
            k = with_sharding_constraint(k, PS(("dp", "fsdp"), None, "mp", None))
            v = with_sharding_constraint(v, PS(("dp", "fsdp"), None, "mp", None))

            # Compute attention scores for this block
            scores = jnp.einsum('bqhd,bkhd->bhqk', q, k)
            scores = with_sharding_constraint(scores, PS(("dp", "fsdp"), "mp", None, None))

            from EasyLM.jax_utils import debug_tensor, create_debug_gather_fn
            # Create gather functions with appropriate partition specs
            query_gather_fn = create_debug_gather_fn(partition_spec=PS(("dp", "fsdp"), None, "mp", None))
            scores_gather_fn = create_debug_gather_fn(partition_spec=PS(("dp", "fsdp"), "mp", None, None))
            
            debug_tensor("Query", q, gather_fn=query_gather_fn)
            debug_tensor("Key", k, gather_fn=query_gather_fn)  # Same spec as query
            debug_tensor("Raw scores", scores, gather_fn=scores_gather_fn)

            if bias is not None:
                # Handle GQA bias if provided
                if bias.shape[1] == num_q_heads:
                    bias_block = jax.lax.dynamic_slice(
                        bias,
                        (0, 0, idx_n * chunk_size, idx_k * chunk_size),
                        (bias.shape[0], bias.shape[1], chunk_size, chunk_size)
                    )
                elif bias.shape[1] == num_kv_heads:
                    bias_block = jax.lax.dynamic_slice(
                        bias,
                        (0, 0, idx_n * chunk_size, idx_k * chunk_size),
                        (bias.shape[0], bias.shape[1], chunk_size, chunk_size)
                    )
                    bias_block = einops.repeat(
                        bias_block, 'b h q k -> b (h g) q k', g=num_groups
                    )
                scores = scores + bias_block

            if causal:
                q_pos = idx_n * chunk_size + jnp.arange(chunk_size)
                k_pos = idx_k * chunk_size + jnp.arange(chunk_size)
                causal_mask = q_pos[:, None] < k_pos[None, :]
                scores = jnp.where(
                    causal_mask[None, None, :, :],
                    jnp.finfo(scores.dtype).min,
                    scores
                )

            m_new = jnp.maximum(m_inner, scores.max(-1, keepdims=True))
            scores = jnp.exp(scores - m_new)
            scores_gather_fn = create_debug_gather_fn(partition_spec=PS(("dp", "fsdp"), "mp", None, None))
            debug_tensor("Post-softmax scores", scores, gather_fn=scores_gather_fn)
            l_new = l_inner * jnp.exp(m_inner - m_new) + scores.sum(-1, keepdims=True)
            # Reshape m_new for proper broadcasting
            scale = jnp.exp(m_inner - m_new)  # [batch, num_heads, chunk_size, 1]
            # Reshape scale to match o_inner dimensions
            scale = jnp.expand_dims(scale, axis=-1)  # Add dim for head_dim
            scale = jnp.transpose(scale, (0, 2, 1, 3, 4))  # Reorder to (batch, num_heads, chunk_size, 1, 1)

            o_new = (o_inner * scale[..., 0] +  # Remove last dim after broadcast
                     jnp.einsum('bhqk,bkhd->bqhd', scores, v))

            # Apply sharding to intermediate results
            o_new = with_sharding_constraint(o_new, PS(("dp", "fsdp"), None, "mp", None))

            return (m_new, l_new, o_new), None

        # Debug the scan length
        jax.debug.print("Inner scan length: {length}", length=key.shape[1])
        
        # Scan over key/value chunks
        (m_new, l_new, o_new), _ = jax.lax.scan(
            lambda carry, idx: kv_chunk_scanner(*carry, idx, idx_n),
            (m, l, o),
            jnp.arange(key.shape[1])
        )

        return (m_new, l_new, o_new), o_new

    # Debug prints to verify scan setup
    jax.debug.print("Query shape before scan: {shape}", shape=query.shape)
    jax.debug.print("Number of chunks to process: {n}", n=query.shape[1])
    jax.debug.print("Chunk size: {size}", size=chunk_size)
    
    def process_query_chunk(idx_n):
        # Get current query chunk
        q = query[:, idx_n]  # [batch, chunk_size, num_q_heads, head_dim]
        q = with_sharding_constraint(q, PS(("dp", "fsdp"), None, "mp", None))

        # Initialize fresh partial sums for this query chunk
        m_init = with_sharding_constraint(
            jnp.full((batch_size, num_q_heads, chunk_size, 1), -jnp.inf),
            PS(("dp", "fsdp"), "mp", None, None)
        )
        l_init = with_sharding_constraint(
            jnp.zeros((batch_size, num_q_heads, chunk_size, 1)),
            PS(("dp", "fsdp"), "mp", None, None)
        )
        o_init = with_sharding_constraint(
            jnp.zeros((batch_size, chunk_size, num_q_heads, head_dim)),
            PS(("dp", "fsdp"), None, "mp", None)
        )

        # Scan over key chunks with fresh partial sums
        (_, _, o_final), _ = jax.lax.scan(
            lambda carry, idx_k: kv_chunk_scanner(q, *carry, idx_k, idx_n),
            (m_init, l_init, o_init),
            jnp.arange(key.shape[1])
        )
        return o_final

    # Process each query chunk independently
    outputs = jax.lax.map(process_query_chunk, jnp.arange(query.shape[1]))
    # outputs shape: [n_chunks, batch, chunk_size, num_heads, head_dim]
    
    # Recombine chunks into final sequence
    output = einops.rearrange(outputs, 'n b c h d -> b (n c) h d')

    # Reshape output back to original sequence length and maintain sharding
    output = einops.rearrange(output, 'n b c h d -> b (n c) h d')
    output = with_sharding_constraint(output, PS(("dp", "fsdp"), None, "mp", None))

    # Debug prints after all operations complete
    from EasyLM.jax_utils import debug_tensor, create_debug_gather_fn
    output_gather_fn = create_debug_gather_fn(partition_spec=PS(("dp", "fsdp"), None, "mp", None))
    debug_tensor("Final output", output, gather_fn=output_gather_fn)

    return output
