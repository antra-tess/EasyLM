import jax
import jax.numpy as jnp
import einops
from jax.sharding import PartitionSpec as PS
from EasyLM.jax_utils import with_sharding_constraint, debug_tensor, create_debug_gather_fn


def flash_attention_2d_blocked(
        query: jnp.ndarray,  # [batch, seq_len, num_q_heads, head_dim]
        key: jnp.ndarray,  # [batch, seq_len, num_kv_heads, head_dim]
        value: jnp.ndarray,  # [batch, seq_len, num_kv_heads, head_dim]
        bias: jnp.ndarray = None,
        causal: bool = True,
        q_chunk_size: int = 128,  # chunk size for queries
        k_chunk_size: int = 128,  # chunk size for keys
):
    """
    Fully 2D-blocked flash attention with row-wise stable softmax.

    * We chunk queries and keys simultaneously.
    * For each block [q_chunk, k_chunk], we maintain a separate
      stable-softmax offset (m_inner) and sum (l_inner) for each row
      in that q_chunk.

    NOTE: This version demonstrates the blocking logic; for extremely large
    models on v4-128, you would add appropriate pjit/with_sharding_constraint
    calls in each step, matching your desired partitions.
    """
    jax.debug.print("Running fully 2D-blocked flash attention with row-wise stability.")

    B, S, QH, D = query.shape
    _, _, KH, _ = key.shape

    # GQA expansion (if QH != KH)
    num_groups = QH // KH
    if (QH % KH) != 0:
        raise ValueError("num_q_heads must be a multiple of num_kv_heads.")

    # Scale the query
    scale = jnp.sqrt(D).astype(query.dtype)
    query = query / scale

    # Reshape into chunks along sequence
    # [batch, n_q_chunks, q_chunk_size, q_heads, dim]
    query = einops.rearrange(
        query,
        'b (nq qc) h d -> b nq qc h d',
        qc=q_chunk_size
    )
    # [batch, n_k_chunks, k_chunk_size, k_heads, dim]
    key = einops.rearrange(
        key,
        'b (nk kc) h d -> b nk kc h d',
        kc=k_chunk_size
    )
    value = einops.rearrange(
        value,
        'b (nk kc) h d -> b nk kc h d',
        kc=k_chunk_size
    )

    # OPTIONAL: Add sharding constraints per-chunk dimension.
    query = with_sharding_constraint(
        query, PS(("dp", "fsdp"), None, None, "mp", None)
    )
    key = with_sharding_constraint(
        key, PS(("dp", "fsdp"), None, None, "mp", None)
    )
    value = with_sharding_constraint(
        value, PS(("dp", "fsdp"), None, None, "mp", None)
    )

    n_q_chunks = query.shape[1]
    n_k_chunks = key.shape[1]

    # Prepare output accumulator. We will fill it chunk-by-chunk.
    # We'll build a list of partial results for each q-chunk,
    # then stack them at the end.
    out_chunks = []

    # ----------------------------------------
    # Outer loop over each Q-chunk
    # ----------------------------------------
    for nq in range(n_q_chunks):
        # Extract the query block: shape [B, q_chunk_size, QH, D]
        q_block = query[:, nq]  # [B, qc, QH, D]

        # We'll build up O_block in a stable way, scanning over k-chunks
        #   m_inner, l_inner shape = [B, QH, qc, 1]
        #   o_inner shape = [B, qc, QH, D]
        m_inner = jnp.full(
            (B, QH, q_chunk_size, 1),
            -jnp.inf,
            dtype=query.dtype
        )
        l_inner = jnp.zeros(
            (B, QH, q_chunk_size, 1),
            dtype=query.dtype
        )
        o_inner = jnp.zeros(
            (B, q_chunk_size, QH, D),
            dtype=query.dtype
        )

        # ----------------------------------------
        # Inner loop over each K-chunk
        # ----------------------------------------
        for nk in range(n_k_chunks):
            # k_block: [B, k_chunk_size, KH, D]
            # v_block: [B, k_chunk_size, KH, D]
            k_block = key[:, nk]
            v_block = value[:, nk]

            # Expand key/value heads for GQA:
            #   repeated across 'num_groups' => shape [B, k_chunk_size, QH, D]
            k_block = einops.repeat(
                k_block,
                'b kc kh d -> b kc (kh g) d',
                g=num_groups
            )
            v_block = einops.repeat(
                v_block,
                'b kc kh d -> b kc (kh g) d',
                g=num_groups
            )

            # *** CHANGED ***: Must keep shapes consistent
            #  q_block is [B, qc, QH, D]
            #  k_block is [B, kc, QH, D]
            # => scores shape [B, QH, qc, kc]
            scores = jnp.einsum('bqhd,bkhd->bhqk', q_block, k_block)
            # Dimension order is [batch, heads, q_chunk, k_chunk]

            # *** ADDED ***: If we have a bias, slice out the relevant portion
            # for [nq, nk] blocks, and broadcast for GQA if needed.
            if bias is not None:
                # shape of 'bias' expected: [B, KH or QH, S, S] or something similar
                # We'll do a quick dynamic_slice for the sub-block:
                q_start = nq * q_chunk_size
                k_start = nk * k_chunk_size
                q_size = q_chunk_size
                k_size = k_chunk_size

                bias_block = jax.lax.dynamic_slice(
                    bias,
                    (0, 0, q_start, k_start),
                    (bias.shape[0], bias.shape[1], q_size, k_size)
                )
                # If bias has only KH heads, repeat for GQA:
                if bias.shape[1] == KH:
                    bias_block = einops.repeat(
                        bias_block, 'b kh q k -> b (kh g) q k', g=num_groups
                    )
                # Now shape matches [B, QH, qc, kc]
                scores = scores + bias_block

            # *** CHANGED ***: row-wise stable softmax
            # `scores` is [B, QH, qc, kc]
            # Debug raw attention scores with more detail
            jax.debug.print("\n=== Processing block q={q} k={k} ===", q=idx_n, k=idx_k)
            jax.debug.print("Raw scores overall - Mean: {}, Max: {}, Min: {}", 
                           jnp.mean(scores), jnp.max(scores), jnp.min(scores))
            
            # Debug per-position scores in this block
            for pos in range(min(scores.shape[2], 4)):  # First 4 positions
                pos_scores = scores[:, :, pos, :]
                jax.debug.print("Position {pos} scores - Mean: {}, Max: {}, Min: {}", 
                              pos=pos + idx_n * scores.shape[2],
                              jnp.mean(pos_scores), jnp.max(pos_scores), jnp.min(pos_scores))

            # We want to update each query row's running max in `m_inner`.
            # `m_inner` is [B, QH, qc, 1], so we do a max over last dim (k).
            # That yields shape [B, QH, qc, 1].
            max_block = jnp.max(scores, axis=-1, keepdims=True)  # per row
            m_new = jnp.maximum(m_inner, max_block)  # shape [B, QH, qc, 1]

            # Debug max values to verify per-query stability
            jax.debug.print("Max values per query chunk - Max: {}, Min: {}", 
                           jnp.max(m_new), jnp.min(m_new))

            # Exponentiate shifted by the new max
            scores = jnp.exp(scores - m_new)  # broadcast sub per row

            # Debug softmax outputs
            jax.debug.print("Post-softmax stats - Mean: {}, Max: {}, Min: {}", 
                           jnp.mean(scores), jnp.max(scores), jnp.min(scores))

            # *** CHANGED ***: accumulate the stable sums
            # l_inner is [B, QH, qc, 1]
            # l_new = l_inner * exp(m_inner - m_new) + sum(exp(scores))
            exp_factor = jnp.exp(m_inner - m_new)  # [B, QH, qc, 1]
            sum_scores = jnp.sum(scores, axis=-1, keepdims=True)  # [B, QH, qc, 1]
            l_new = l_inner * exp_factor + sum_scores

            # *** ADDED ***: Apply causal if needed
            if causal:
                # We need to figure out the global offset for query vs. key
                #   e.g. q positions = [nq * q_chunk_size ... (nq+1)*q_chunk_size-1]
                #   same for k.  Then build a mask that zeros out future keys.
                # shape of `scores`: [B, QH, qc, kc]
                q_positions = jnp.arange(q_chunk_size) + (nq * q_chunk_size)
                k_positions = jnp.arange(k_chunk_size) + (nk * k_chunk_size)
                # mask has shape [qc, kc], where 1 => "disallowed"
                causal_mask = q_positions[:, None] < k_positions[None, :]
                # Expand to [1, 1, qc, kc] for broadcast
                causal_mask = causal_mask[None, None, :, :]

                # Now we must reapply the “-∞” logic *before* exponentiation,
                # or effectively zero out those illegal positions after.
                # If we do it *after* exponent, we set those terms to 0.0:
                scores = scores * (1.0 - causal_mask.astype(scores.dtype))

                # Because we already computed the max shift, we also must fix
                # partial sums in l_new for those positions. Easiest to subtract:
                blocked_sum = jnp.sum(scores * causal_mask, axis=-1, keepdims=True)
                l_new = l_new - blocked_sum

            # *** CHANGED ***: Recompute final partial sums
            # since we possibly applied the causal mask after exponent.
            # sum_scores might need an update if causal cut out some keys:
            sum_scores = jnp.sum(scores, axis=-1, keepdims=True)  # [B, QH, qc, 1]
            # Then l_new = (old l)*exp_factor + sum_scores
            # but we have to do it carefully if we subtracted blocked_sum above.
            # For clarity, re-assign:
            l_new = l_inner * exp_factor + sum_scores

            # *** CHANGED ***: update the running output
            # o_inner is [B, qc, QH, D]
            # we have scores shape [B, QH, qc, kc]
            # v_block shape [B, kc, QH, D]
            # So let's compute partial attn * V
            # We'll do einops to put Q dimension last or match the above.
            # Let's reorder 'scores' to [B, QH, qc, kc] => 'bhqk'
            # Then v_block is [B, kc, QH, D] => we want [B, QH, kc, D] for easier einsum
            v_block_t = einops.rearrange(v_block, 'b kc h d -> b h kc d')
            delta_o = jnp.einsum('bhqk, bhkd->bqhd', scores, v_block_t)
            # shape of delta_o is [B, qc, QH, D]

            # We must scale the old o_inner by exp_factor:
            #   o_inner_new = o_inner * exp_factor + delta_o
            # But exp_factor shape is [B, QH, qc, 1],
            # so we reorder it to broadcast along [B, qc, QH, D].
            exp_factor_b = einops.rearrange(exp_factor, 'b h qc 1 -> b qc h 1')
            o_new = o_inner * exp_factor_b + delta_o

            # *** DONE ***: store updated partial results
            m_inner = m_new
            l_inner = l_new
            o_inner = o_new

        # Done scanning over all k-chunks -> normalize final output block
        # l_inner shape [B, QH, qc, 1], reorder for broadcast
        l_bcast = einops.rearrange(l_inner, 'b h qc 1 -> b qc h 1')
        jax.debug.print("Normalization factor stats - Mean: {}, Max: {}, Min: {}", 
                       jnp.mean(l_bcast), jnp.max(l_bcast), jnp.min(l_bcast))
        
        # o_inner: [B, qc, QH, D]
        o_block = o_inner / (l_bcast + 1e-9)
        jax.debug.print("Output block stats - Mean: {}, Max: {}, Min: {}", 
                       jnp.mean(o_block), jnp.max(o_block), jnp.min(o_block))
        
        # Debug first/middle/last token in block
        jax.debug.print("Block tokens [first, middle, last]: [{}, {}, {}]",
                       o_block[0, 0, 0, 0], 
                       o_block[0, o_block.shape[1]//2, 0, 0],
                       o_block[0, -1, 0, 0])
        
        out_chunks.append(o_block)

    # Combine all q-chunks back
    # out_chunks is a list of length n_q_chunks,
    # each shape [B, q_chunk_size, QH, D]
    output = jnp.concatenate(out_chunks, axis=1)  # reassemble along seq dimension
    jax.debug.print("Combined output shape: {}", output.shape)
    
    # Debug final output statistics
    jax.debug.print("Final output stats - Mean: {}, Max: {}, Min: {}", 
                   jnp.mean(output), jnp.max(output), jnp.min(output))
    
    # Debug attention patterns across sequence
    jax.debug.print("Attention pattern check:")
    jax.debug.print("First token outputs - Mean: {}, Max: {}, Min: {}", 
                   jnp.mean(output[:, 0]), jnp.max(output[:, 0]), jnp.min(output[:, 0]))
    jax.debug.print("Middle token outputs - Mean: {}, Max: {}, Min: {}", 
                   jnp.mean(output[:, output.shape[1]//2]), 
                   jnp.max(output[:, output.shape[1]//2]), 
                   jnp.min(output[:, output.shape[1]//2]))
    jax.debug.print("Last token outputs - Mean: {}, Max: {}, Min: {}", 
                   jnp.mean(output[:, -1]), jnp.max(output[:, -1]), jnp.min(output[:, -1]))

    # Finally, restore shape [batch, seq_len, QH, D]
    output = with_sharding_constraint(
        output, PS(("dp", "fsdp"), None, "mp", None)
    )
    return output
