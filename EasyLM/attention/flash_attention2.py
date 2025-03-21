import logging
from functools import partial

import jax
import jax.numpy as jnp
import einops
from typing import Optional
from EasyLM.jax_utils import debug_tensor

from jax.sharding import PartitionSpec as PS
from jax.experimental.pjit import pjit
from EasyLM.jax_utils import (
    with_sharding_constraint,
    create_debug_gather_fn,
    debug_tensor,
)


def flash_attention_2d_blocked(
    query: jnp.ndarray,    # [batch, seq_len, num_q_heads, head_dim]
    key: jnp.ndarray,      # [batch, seq_len, num_kv_heads, head_dim]
    value: jnp.ndarray,    # [batch, seq_len, num_kv_heads, head_dim]
    bias: Optional[jnp.ndarray] = None,
    causal: bool = True,
    q_chunk_size: int = 128,
    kv_chunk_size: int = 128,
) -> jnp.ndarray:
    """
    2D-blocked Flash Attention with GQA and stable softmax.

    This implementation partitions the input into (q_chunk, k_chunk) blocks
    and does a row-wise stable softmax accumulation for each block. GQA is
    handled by repeating the key/value heads as needed.

    The final shape is [batch, seq_len, num_q_heads, head_dim].

    Sharding assumptions (adjust as needed for your environment):
      - query, key, value: PS(("dp", "fsdp"), None, "mp", None)
        i.e. batch+fsdp on the first dimension, head-sharding on the 3rd dimension, etc.

    Args:
        query:    [batch, seq_len, num_q_heads, head_dim]
        key:      [batch, seq_len, num_kv_heads, head_dim]
        value:    [batch, seq_len, num_kv_heads, head_dim]
        bias:     [batch, #heads, seq_len, seq_len], if not None
        causal:   Whether to apply causal masking
        q_chunk_size: How many query tokens per chunk
        kv_chunk_size: How many key tokens per chunk

    Returns:
        [batch, seq_len, num_q_heads, head_dim]
    """
    jax.debug.print("Starting 2D-blocked Flash Attention")

    # Debug input tensors
    from EasyLM.jax_utils import debug_tensor, create_debug_gather_fn
    query_gather_fn = create_debug_gather_fn(partition_spec=PS(("dp", "fsdp"), None, "mp", None))
    key_gather_fn = create_debug_gather_fn(partition_spec=PS(("dp", "fsdp"), None, "mp", None))
    value_gather_fn = create_debug_gather_fn(partition_spec=PS(("dp", "fsdp"), None, "mp", None))

    debug_tensor("Initial query", query, gather_fn=query_gather_fn)
    debug_tensor("Initial key", key, gather_fn=key_gather_fn)
    debug_tensor("Initial value", value, gather_fn=value_gather_fn)

    batch_size, seq_len, num_q_heads, head_dim = query.shape
    _, _, num_kv_heads, _ = key.shape

    # GQA factor
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    num_groups = num_q_heads // num_kv_heads

    # Precompute token positions [0, 1, 2, ..., seq_len-1]
    all_positions = jnp.arange(seq_len, dtype=jnp.int32)

    # Scale the queries by sqrt(head_dim)
    scale = jnp.sqrt(jnp.array(head_dim, dtype=query.dtype))
    query = query / scale

    # Helper function for dynamic slicing of key/value blocks
    def get_kv_chunk(blocked_array, k_idx):
        # k_idx is traced integer. We slice axis=1 (the n_k dimension)
        chunk = jax.lax.dynamic_slice_in_dim(
            blocked_array,
            start_index=k_idx,
            slice_size=1,
            axis=1
        )
        # Squeeze out the block dimension: [b, 1, kc, heads, d] -> [b, kc, heads, d]
        return jnp.squeeze(chunk, axis=1)

    # Reshape the sequences into 2D blocks: [b, n_q, q_chunk, heads, dim], [b, n_k, k_chunk, heads, dim]
    # where n_q = seq_len // q_chunk_size, n_k = seq_len // kv_chunk_size (assuming divisibility)
    n_q = seq_len // q_chunk_size
    n_k = seq_len // kv_chunk_size
    query_blocked = einops.rearrange(
        query, "b (nq qc) h d -> b nq qc h d", qc=q_chunk_size
    )
    key_blocked = einops.rearrange(
        key, "b (nk kc) h d -> b nk kc h d", kc=kv_chunk_size
    )
    value_blocked = einops.rearrange(
        value, "b (nk kc) h d -> b nk kc h d", kc=kv_chunk_size
    )

    # Apply sharding constraints after the reshape
    query_blocked = with_sharding_constraint(
        query_blocked, PS(("dp", "fsdp"), None, None, "mp", None)
    )
    key_blocked = with_sharding_constraint(
        key_blocked, PS(("dp", "fsdp"), None, None, "mp", None)
    )
    value_blocked = with_sharding_constraint(
        value_blocked, PS(("dp", "fsdp"), None, None, "mp", None)
    )

    # We'll accumulate the final outputs in a similar blocked layout, then rearrange back
    # shape -> [b, nq, qc, num_q_heads, head_dim]
    # We'll build them up chunk by chunk using a scan or a double for-loop in Python

    def compute_one_q_chunk(q_chunk: jnp.ndarray, q_block_idx: int):
        """
        For a single chunk of queries of shape [batch, q_chunk_size, num_q_heads, head_dim],
        accumulate over all key-chunks in a stable manner.
        """
        # q_chunk: [b, qc, q_heads, d]
        # We'll track partial sums with shape [b, h, q, 1] for m and l
        # and [b, h, q, d] for o to match bhqk layout
        m_init = jnp.full((batch_size, q_chunk_size, num_q_heads, 1), -jnp.inf, dtype=q_chunk.dtype)
        l_init = jnp.zeros((batch_size, q_chunk_size, num_q_heads, 1), dtype=q_chunk.dtype)
        o_init = jnp.zeros((batch_size, q_chunk_size, num_q_heads, head_dim), dtype=q_chunk.dtype)

        m_init = with_sharding_constraint(m_init, PS(("dp", "fsdp"), "mp", None, None))
        l_init = with_sharding_constraint(l_init, PS(("dp", "fsdp"), "mp", None, None))
        o_init = with_sharding_constraint(o_init, PS(("dp", "fsdp"), "mp", None, None))

        def attend_to_k_chunk(carry, k_block_idx: int):
            """
            Accumulate attention from the k-th key-block into the partial sums.
            carry = (m, l, o)
            k_block_idx is an integer in [0, n_k).
            """
            (m_curr, l_curr, o_curr) = carry

            # Slice out the actual chunk of K/V using dynamic slice
            k_chunk = get_kv_chunk(key_blocked, k_block_idx)  # [batch, k_chunk_size, num_kv_heads, head_dim]
            v_chunk = get_kv_chunk(value_blocked, k_block_idx)

            # Debug input values
            debug_tensor(f"Key chunk (k={k_block_idx})", k_chunk)
            debug_tensor(f"Value chunk (k={k_block_idx})", v_chunk)

            # Get correct position offsets for this block using dynamic_slice
            k_offset = k_block_idx * kv_chunk_size
            local_k_positions = jax.lax.dynamic_slice(
                all_positions,
                (k_offset,),
                (kv_chunk_size,)
            )

            # GQA: expand key/value heads to match q_heads = num_kv_heads * num_groups
            k_chunk = einops.repeat(
                k_chunk,
                "b kc h d -> b kc (h g) d",
                g=num_groups,
            )
            v_chunk = einops.repeat(
                v_chunk,
                "b kc h d -> b kc (h g) d",
                g=num_groups,
            )

            # Shard after repeating
            k_chunk = with_sharding_constraint(
                k_chunk, PS(("dp", "fsdp"), None, "mp", None)
            )
            v_chunk = with_sharding_constraint(
                v_chunk, PS(("dp", "fsdp"), None, "mp", None)
            )

            # Compute raw scores:
            # q_chunk [b, qc, heads, d] x k_chunk [b, kc, heads, d]
            # -> [b, heads, qc, kc] for consistent bhqk layout
            scores = jnp.einsum("bqhd,bkhd->bqhk", q_chunk, k_chunk)
            scores = with_sharding_constraint(scores, PS(("dp", "fsdp"), None, "mp", None))

            # Optionally add bias
            # shape of bias: [b, #heads, seq_len, seq_len]
            # We need to slice out [qc, kc] for these chunk indices
            if bias is not None:
                # Indices in the original sequence
                q_offset = q_block_idx * q_chunk_size
                k_offset = k_block_idx * kv_chunk_size

                # In GQA, bias might be shaped for kv_heads vs q_heads. We handle accordingly:
                if bias.shape[1] == num_q_heads:
                    bias_block = jax.lax.dynamic_slice(
                        bias,
                        (0, 0, q_offset, k_offset),
                        (batch_size, num_q_heads, q_chunk_size, kv_chunk_size),
                    )
                    # Transpose bias to match bqhk layout
                    bias_block = jnp.transpose(bias_block, (0, 2, 1, 3))
                elif bias.shape[1] == num_kv_heads:
                    bias_block = jax.lax.dynamic_slice(
                        bias,
                        (0, 0, q_offset, k_offset),
                        (batch_size, num_kv_heads, q_chunk_size, kv_chunk_size),
                    )
                    bias_block = einops.repeat(
                        bias_block, "b h q k -> b q (h g) k", g=num_groups
                    )
                else:
                    raise ValueError("Unsupported bias shape for GQA")

                scores = scores + bias_block

            # Compute raw scores and apply causal masking if needed
            debug_tensor(f"Raw scores before mask (q={q_block_idx}, k={k_block_idx})", scores)

            if causal:
                # Get query positions for this block using dynamic_slice
                q_offset = q_block_idx * q_chunk_size
                local_q_positions = jax.lax.dynamic_slice(
                    all_positions,
                    (q_offset,),
                    (q_chunk_size,)
                )
                
                # Build causal mask using correct absolute positions
                # shape [qc, kc] where True means position should be masked
                causal_mask = local_q_positions[:, None] < local_k_positions[None, :]  # Mask future positions
                # Expand mask to [1, qc, 1, kc] for broadcasting with [b, qc, h, kc]
                causal_mask = causal_mask[None, :, None, :]
                scores = jnp.where(
                    causal_mask,
                    jnp.finfo(scores.dtype).min,
                    scores,
                )
                
                debug_tensor(f"Scores after causal mask (q={q_block_idx}, k={k_block_idx})", scores)

            # Now proceed with stable softmax using global max
            block_max = jnp.max(scores, axis=-1, keepdims=True)   # shape [b,q,h,1]
            debug_tensor(f"Block max (q={q_block_idx}, k={k_block_idx})", block_max)

            # 2) Update global max - this is what we'll use for shifting
            m_new = jnp.maximum(m_curr, block_max)
            debug_tensor(f"Global max (q={q_block_idx}, k={k_block_idx})", m_new)

            # 3) Exponentiate scores using global max m_new (not local block_max)
            # Add numerical stability check
            shifted = scores - m_new
            scores_shifted = jnp.where(
                shifted < -10.0,  # threshold for numerical stability
                jnp.zeros_like(shifted),
                jnp.exp(shifted)
            )
            debug_tensor(f"Shifted scores (q={q_block_idx}, k={k_block_idx})", scores_shifted)

            # 4) Scale old partial sums by exp_factor = exp(m_curr - m_new)
            exp_factor = jnp.exp(m_curr - m_new)
            
            # Update running sums
            sum_scores = jnp.sum(scores_shifted, axis=-1, keepdims=True)
            l_new = l_curr * exp_factor + sum_scores  # [b, qc, heads, 1]

            # Since scores_shifted is [b, q, h, k] and v_chunk is [b, k, h, d],
            # we want out_block in [b, h, q, d] to match bhqk layout
            out_block = jnp.einsum("bqhk,bkhd->bqhd", scores_shifted, v_chunk)


            # We must multiply existing o_curr by exp_factor, which is shaped [b, qc, heads, 1]
            o_new = o_curr * exp_factor + out_block

            # Shard each updated partial
            m_new = with_sharding_constraint(m_new, PS(("dp", "fsdp"), None, "mp", None))
            l_new = with_sharding_constraint(l_new, PS(("dp", "fsdp"), None, "mp", None))
            o_new = with_sharding_constraint(o_new, PS(("dp", "fsdp"), None, "mp", None))

            # Debug normalization factors
            debug_tensor(f"exp_factor (q={q_block_idx}, k={k_block_idx})", exp_factor)
            debug_tensor(f"sum_scores before norm (q={q_block_idx}, k={k_block_idx})", sum_scores)
            debug_tensor(f"l_new (denominator) (q={q_block_idx}, k={k_block_idx})", l_new)
            
            # Debug intermediate output
            debug_tensor(f"out_block before scaling (q={q_block_idx}, k={k_block_idx})", out_block)
            debug_tensor(f"o_new (numerator) (q={q_block_idx}, k={k_block_idx})", o_new)

            return (m_new, l_new, o_new), None

        # Now do a scan over the key-chunks for this single q-chunk
        (m_final, l_final, o_final), _ = jax.lax.scan(
            attend_to_k_chunk, (m_init, l_init, o_init), jnp.arange(n_k)
        )

        # Normalize final o_final by l_final
        # shapes: o_final [b, qc, heads, d], l_final [b, qc, heads, 1]
        o_normalized = o_final / l_final

        # Debug final values for this q_chunk
        debug_tensor(f"Final l (denominator) for q_chunk {q_block_idx}", l_final)
        debug_tensor(f"Final o (numerator) for q_chunk {q_block_idx}", o_final)
        debug_tensor(f"Normalized output for q_chunk {q_block_idx}", o_normalized)

        return o_normalized

    def process_all_q_chunks(q_block_idx: int):
        # We pick out the q-th chunk
        q_chunk = query_blocked[:, q_block_idx]  # [b, q_chunk_size, num_q_heads, d]
        q_chunk = with_sharding_constraint(q_chunk, PS(("dp", "fsdp"), None, "mp", None))

        # Accumulate over all key-chunks
        return compute_one_q_chunk(q_chunk, q_block_idx)

    # Map over all q-blocks. This yields shape [n_q, b, qc, heads, d]
    all_q_outputs = jax.lax.map(process_all_q_chunks, jnp.arange(n_q))

    debug_tensor("Final output blocks", all_q_outputs)
    output = einops.rearrange(
            all_q_outputs, "nq b qc h d -> b (nq qc) h d",
            nq=n_q, qc=q_chunk_size
    )

    # Apply final sharding constraint
    output = with_sharding_constraint(output, PS(("dp", "fsdp"), None, "mp", None))

    # Debug final reshaped output
    debug_tensor("Final reshaped output", output)

    return output
