import jax
import jax.numpy as jnp
import einops
from typing import Optional
from jax.sharding import PartitionSpec as PS
from EasyLM.jax_utils import with_sharding_constraint
from jax.experimental.multihost_utils import process_allgather


def flash_attention(
    query: jnp.ndarray,  # [batch, seq_len, num_q_heads, head_dim]
    key: jnp.ndarray,    # [batch, seq_len, num_kv_heads, head_dim] 
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

    @jax.jit
    def chunk_scanner(carry, idx_n):
        m, l, o = carry  # max_so_far, l_acc, output_acc
        
        # Get current query chunk and maintain sharding
        q = query[:, idx_n]  # [batch, chunk_size, num_q_heads, head_dim]
        q = with_sharding_constraint(q, PS(("dp", "fsdp"), None, "mp", None))
        
        @jax.jit
        def kv_chunk_scanner(carry, idx_k):
            m_inner, l_inner, o_inner = carry
            
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
            
            # Compute attention scores for this block and debug
            scores = jnp.einsum('bqhd,bkhd->bhqk', q, k)
            scores = with_sharding_constraint(scores, PS(("dp", "fsdp"), "mp", None, None))
            
            # Compute attention scores for this block
            scores = jnp.einsum('bqhd,bkhd->bhqk', q, k)
            scores = with_sharding_constraint(scores, PS(("dp", "fsdp"), "mp", None, None))
            
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
                # Pre-compute positions using static arrays
                with jax.ensure_compile_time_eval():
                    q_pos = jnp.arange(chunk_size)
                    k_pos = jnp.arange(chunk_size)
                    causal_mask = (idx_n * chunk_size + q_pos[:, None]) < (idx_k * chunk_size + k_pos[None, :])
                
                scores = jnp.where(
                    causal_mask[None, None, :, :],
                    jnp.finfo(scores.dtype).min,
                    scores
                )

            m_new = jnp.maximum(m_inner, scores.max(-1, keepdims=True))
            scores = jnp.exp(scores - m_new)
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

        # Use static indices for inner scan
        num_kv_chunks = key.shape[1]
        kv_indices = jax.lax.iota(jnp.int32, num_kv_chunks)
        
        # Scan over key/value chunks
        (m_new, l_new, o_new), _ = jax.lax.scan(
            kv_chunk_scanner,
            (m, l, o),
            kv_indices
        )
        
        return (m_new, l_new, o_new), o_new

    # Initialize carry with appropriate sharding
    init_m = with_sharding_constraint(
        jnp.full((batch_size, num_q_heads, chunk_size, 1), -jnp.inf),
        PS(("dp", "fsdp"), "mp", None, None)
    )
    init_l = with_sharding_constraint(
        jnp.zeros((batch_size, num_q_heads, chunk_size, 1)),
        PS(("dp", "fsdp"), "mp", None, None)
    )
    init_o = with_sharding_constraint(
        jnp.zeros((batch_size, chunk_size, num_q_heads, head_dim)),
        PS(("dp", "fsdp"), None, "mp", None)
    )
    
    # Use static indices for scan
    num_chunks = query.shape[1]
    indices = jax.lax.iota(jnp.int32, num_chunks)
    
    # Scan over query chunks
    _, output = jax.lax.scan(
        chunk_scanner,
        (init_m, init_l, init_o),
        indices
    )
    
    # Reshape output back to original sequence length and maintain sharding
    output = einops.rearrange(output, 'n b c h d -> b (n c) h d')
    output = with_sharding_constraint(output, PS(("dp", "fsdp"), None, "mp", None))
    
    # Reshape output back to original sequence length and maintain sharding
    output = einops.rearrange(output, 'n b c h d -> b (n c) h d')
    output = with_sharding_constraint(output, PS(("dp", "fsdp"), None, "mp", None))
    
    # # Debug prints after all operations complete
    # output_gathered = process_allgather(output)
    # jax.debug.print("Output shape: {shape}", shape=output_gathered.shape)
    # jax.debug.print("First token values: {values}", values=output_gathered[0, :4, 0, 0])
    
    return output
