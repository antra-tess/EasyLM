import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as PS
from absl.testing import absltest, parameterized
from EasyLM.jax_utils import with_sharding_constraint, get_jax_mesh
from jax.experimental.multihost_utils import process_allgather

from EasyLM.attention.flash_attention2 import flash_attention_2d_blocked as flash_attention


class FlashAttentionTest(parameterized.TestCase):
    def setUp(self):
        # Set up mesh using LLaMAConfigurator's method
        self.mesh = get_jax_mesh("1,-1,4", ("dp", "fsdp", "mp"))

    def get_attention_inputs(self, batch=32, seq_len=16, num_q_heads=8, num_kv_heads=4, head_dim=32):
        """Helper to create test inputs with proper sharding."""
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)

        # Create inputs
        query = jax.random.normal(k1, (batch, seq_len, num_q_heads, head_dim))
        key_tensor = jax.random.normal(k2, (batch, seq_len, num_kv_heads, head_dim))
        value = jax.random.normal(k3, (batch, seq_len, num_kv_heads, head_dim))

        # Apply sharding using make_array_from_callback
        with self.mesh:
            # Use with_sharding_constraint for multi-host compatibility
            # Use the same partition: (dp, fsdp) on batch, "mp" on heads
            query = with_sharding_constraint(query, PS(("dp", "fsdp"), None, "mp", None))
            key = with_sharding_constraint(key_tensor, PS(("dp", "fsdp"), None, "mp", None))
            value = with_sharding_constraint(value, PS(("dp", "fsdp"), None, "mp", None))

        return query, key, value

    def reference_attention(self, query, key, value, causal=True):
        """Reference implementation for comparison."""
        from EasyLM.jax_utils import debug_tensor

        batch_size, seq_len, num_q_heads, head_dim = query.shape
        _, _, num_kv_heads, _ = key.shape

        # Handle GQA
        num_groups = num_q_heads // num_kv_heads
        key = jnp.repeat(key, num_groups, axis=2)
        value = jnp.repeat(value, num_groups, axis=2)

        # Scale query
        query = query / jnp.sqrt(head_dim)

        # Compute attention scores
        scores = jnp.einsum('bqhd,bkhd->bhqk', query, key)

        if causal:
            mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
            scores = jnp.where(
                mask[None, None, :, :],
                jnp.finfo(scores.dtype).min,
                scores
            )

        # Debug raw scores before softmax
        debug_tensor("Raw scores before softmax", scores)
            
        # Apply softmax per query
        scores = jax.nn.softmax(scores, axis=-1)
            
        # Debug post-softmax scores
        debug_tensor("Post-softmax scores", scores)

        # Compute output
        output = jnp.einsum('bhqk,bkhd->bqhd', scores, value)
        return output


    def test_attention_patterns(self):
        """Test specific attention patterns."""
        with self.mesh:
            # Create a sequence where middle tokens should attend strongly to first token
            batch_size, seq_len, num_heads, head_dim = 32, 4, 4, 32  # Match mesh dimensions: fsdp=8, mp=4
            query = jnp.zeros((batch_size, seq_len, num_heads, head_dim))
            key = value = jnp.zeros((batch_size, seq_len, num_heads, head_dim))

            # First token: strongly negative query so it attends nowhere
            query = query.at[:, 0, :, :].set(-1e9)
            # Middle tokens: strong positive query to attend to first token
            query = query.at[:, 1:3, :, :].set(10.0)
            # First token: strong key for middle tokens to attend to
            key = key.at[:, 0, :, :].set(10.0)
            # First token: value of 1.0 that will be picked up by middle tokens
            value = value.at[:, 0, :, :].set(1.0)



            @jax.jit
            def flash_attention_jit(query, key, value):
                return flash_attention(
                    query=query,
                    key=key,
                    value=value,
                    q_chunk_size=2,
                    kv_chunk_size=2
                )


            # Run flash attention
            output = flash_attention_jit(query, key, value)

            # Middle tokens should get first token's value, others near zero
            expected = jnp.zeros((batch_size, seq_len, num_heads, head_dim))
            expected = expected.at[:, 1:3, :, :].set(1.0)

            # Calculate diff on all processes
            diff = jnp.abs(output - expected)
            max_diff = jnp.max(diff)

            # Gather results for debugging
            output_gathered = process_allgather(output)
            expected_gathered = process_allgather(expected)
            max_diff_idx = jnp.argmax(jnp.abs(output_gathered - expected_gathered))

            jax.debug.print("Output shape: {}", output_gathered.shape)
            jax.debug.print("First token: {}, Middle token: {}, Last token: {}",
                          output_gathered[0, 0, 0, 0], output_gathered[0, 1, 0, 0], output_gathered[0, -1, 0, 0])
            jax.debug.print("Max diff at index {}", max_diff_idx)
            jax.debug.print("Output value at max diff: {}", output_gathered.flatten()[max_diff_idx])
            jax.debug.print("Expected value at max diff: {}", expected_gathered.flatten()[max_diff_idx])

            assert jnp.all(max_diff < 1e-5), f"Attention pattern test failed with max difference {max_diff}"


if __name__ == '__main__':
    absltest.main() 
