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
        # We'll track partial sums: m, l, o for the entire q_chunk
        #   m: [b, qc, q_heads, 1]
        #   l: [b, qc, q_heads, 1]
        #   o: [b, qc, q_heads, d]
        m_init = jnp.full(
            (batch_size, q_chunk_size, num_q_heads, 1),
            -jnp.inf,
            dtype=q_chunk.dtype,
        )
        l_init = jnp.zeros(
            (batch_size, q_chunk_size, num_q_heads, 1), dtype=q_chunk.dtype
        )
        o_init = jnp.zeros(
            (batch_size, q_chunk_size, num_q_heads, head_dim),
            dtype=q_chunk.dtype,
        )

        m_init = with_sharding_constraint(m_init, PS(("dp", "fsdp"), None, "mp", None))
        l_init = with_sharding_constraint(l_init, PS(("dp", "fsdp"), None, "mp", None))
        o_init = with_sharding_constraint(o_init, PS(("dp", "fsdp"), None, "mp", None))

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
            # -> [b, heads, qc, kc]
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
                    # Direct slice
                    bias_block = jax.lax.dynamic_slice(
                        bias,
                        (0, 0, q_offset, k_offset),
                        (batch_size, num_q_heads, q_chunk_size, kv_chunk_size),
                    )
                elif bias.shape[1] == num_kv_heads:
                    # Then repeat it to match q_heads
                    bias_block = jax.lax.dynamic_slice(
                        bias,
                        (0, 0, q_offset, k_offset),
                        (batch_size, num_kv_heads, q_chunk_size, kv_chunk_size),
                    )
                    bias_block = einops.repeat(
                        bias_block, "b h q k -> b (h g) q k", g=num_groups
                    )
                else:
                    raise ValueError("Unsupported bias shape for GQA")

                # Now shape is [b, q_heads, qc, kc]. We match [b, heads, qc, kc].
                scores = scores + bias_block

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
                causal_mask = local_q_positions[:, None] < local_k_positions[None, :]
                scores = jnp.where(
                    causal_mask[None, :, None, :],
                    jnp.finfo(scores.dtype).min,
                    scores,
                )

            # Current block's row-wise max
            # m_curr shape: [b, qc, heads, 1],  scores shape: [b, heads, qc, kc]
            # We want to broadcast each row in qc:
            max_block = jnp.max(scores, axis=-1, keepdims=True)  # [b, heads, qc, 1]
            m_new = jnp.maximum(m_curr, max_block)

            # exponentiate shifted by new max
            scores_shifted = jnp.exp(scores - max_block)
            # old partial sum reweighted
            exp_factor = jnp.exp(m_curr - m_new)
            # l_curr: [b, qc, heads, 1]
            # sum of exponentiated scores: [b, heads, qc, 1] after sum over k
            sum_scores = jnp.sum(scores_shifted, axis=-1, keepdims=True)

            l_new = l_curr * exp_factor + sum_scores  # [b, qc, heads, 1]

            # Since scores_shifted is [b, qc, heads, kc] and v_chunk is [b, kc, heads, d],
            # we want out_block in [b, qc, heads, d].
            # new pattern:    'bqhk,bkh(d)-> bqhd'
            out_block = jnp.einsum("bqhk,bkhd->bqhd", scores_shifted, v_chunk)


            # We must multiply existing o_curr by exp_factor, which is shaped [b, qc, heads, 1]
            o_new = o_curr * exp_factor + out_block

            # Shard each updated partial
            m_new = with_sharding_constraint(m_new, PS(("dp", "fsdp"), None, "mp", None))
            l_new = with_sharding_constraint(l_new, PS(("dp", "fsdp"), None, "mp", None))
            o_new = with_sharding_constraint(o_new, PS(("dp", "fsdp"), None, "mp", None))

            return (m_new, l_new, o_new), None

        # Now do a scan over the key-chunks for this single q-chunk
        (m_final, l_final, o_final), _ = jax.lax.scan(
            attend_to_k_chunk, (m_init, l_init, o_init), jnp.arange(n_k)
        )

        # Normalize final o_final by l_final
        # shapes: o_final [b, qc, heads, d], l_final [b, qc, heads, 1]
        o_normalized = o_final / l_final

        return o_normalized

    def process_all_q_chunks(q_block_idx: int):
        # We pick out the q-th chunk
        q_chunk = query_blocked[:, q_block_idx]  # [b, q_chunk_size, num_q_heads, d]
        q_chunk = with_sharding_constraint(q_chunk, PS(("dp", "fsdp"), None, "mp", None))

        # Accumulate over all key-chunks
        return compute_one_q_chunk(q_chunk, q_block_idx)

    # Map over all q-blocks. This yields shape [n_q, b, qc, heads, d]
    all_q_outputs = jax.lax.map(process_all_q_chunks, jnp.arange(n_q))

    # Rearrange back to [b, (n_q * qc) = seq_len, heads, d]
    output = einops.rearrange(
        all_q_outputs,
        "nq b qc h d -> b (nq qc) h d",
        nq=n_q,
        qc=q_chunk_size
    )

    # Finally, apply a last with_sharding_constraint so the final output has the standard shape
    # consistent with your original: PS(("dp", "fsdp"), None, "mp", None)
    output = with_sharding_constraint(output, PS(("dp", "fsdp"), None, "mp", None))

    return output
