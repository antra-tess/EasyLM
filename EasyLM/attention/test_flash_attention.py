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
        """Test specific attention patterns without causal masking."""
        with self.mesh:
            batch_size, seq_len, num_heads, head_dim = 32, 4, 4, 32
            query = jnp.zeros((batch_size, seq_len, num_heads, head_dim))
            key = value = jnp.zeros((batch_size, seq_len, num_heads, head_dim))

            # Set up a clear attention pattern:
            # - First token: uniform attention (zeros)
            # - Middle tokens (1,2): strong attention to first token (positive values)
            # - Last token: uniform attention (zeros)
            
            # Middle tokens attend strongly to first token
            query = query.at[:, 1:3, :, :].set(1.0)
            # First token has a distinctive key
            key = key.at[:, 0, :, :].set(1.0)
            # First token has value 1.0, others 0.0
            value = value.at[:, 0, :, :].set(1.0)

            # Scale query by 1/sqrt(head_dim) as in real usage
            query = query / jnp.sqrt(head_dim)

            @jax.jit
            def flash_attention_jit(query, key, value):
                return flash_attention(
                    query=query,
                    key=key,
                    value=value,
                    causal=False,  # Test pure attention without masking
                    q_chunk_size=2,
                    kv_chunk_size=2
                )

            output = flash_attention_jit(query, key, value)

            # Expected pattern:
            # - First token: uniform attention -> ~0.25 (average of all values)
            # - Middle tokens: strong attention to first token -> ~1.0
            # - Last token: uniform attention -> ~0.25
            expected = jnp.full((batch_size, seq_len, num_heads, head_dim), 0.25)
            expected = expected.at[:, 1:3, :, :].set(1.0)

            # Calculate diff on all processes
            diff = jnp.abs(output - expected)
            max_diff = jnp.max(diff)

            # Gather results for debugging
            output_gathered = process_allgather(output)
            expected_gathered = process_allgather(expected)

            jax.debug.print("Output shape: {}", output_gathered.shape)
            jax.debug.print("First token (uniform): {}, Middle token (strong): {}, Last token (uniform): {}",
                          output_gathered[0, 0, 0, 0], output_gathered[0, 1, 0, 0], output_gathered[0, -1, 0, 0])

            assert jnp.all(max_diff < 1e-5), f"Attention pattern test failed with max difference {max_diff}"

    # def test_causal_masking(self):
    #     """Test that causal masking works correctly."""
    #     with self.mesh:
    #         batch_size, seq_len, num_heads, head_dim = 32, 4, 4, 32
    #         query = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    #         key = value = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    #
    #         # Scale query by 1/sqrt(head_dim) as in real usage
    #         query = query / jnp.sqrt(head_dim)
    #
    #         @jax.jit
    #         def flash_attention_jit(query, key, value):
    #             return flash_attention(
    #                 query=query,
    #                 key=key,
    #                 value=value,
    #                 causal=True,
    #                 q_chunk_size=2,
    #                 kv_chunk_size=2
    #             )
    #
    #         output = flash_attention_jit(query, key, value)
    #
    #         # With uniform inputs and causal masking:
    #         # - First position (0) should only attend to itself: value[0]
    #         # - Second position (1) should attend to 0,1: mean(value[0:2])
    #         # - Third position (2) should attend to 0,1,2: mean(value[0:3])
    #         # And so on...
    #         expected = jnp.zeros_like(output)
    #         for i in range(seq_len):
    #             expected = expected.at[:, i].set(jnp.mean(value[:, :i+1], axis=1))
    #
    #         diff = jnp.abs(output - expected)
    #         max_diff = jnp.max(diff)
    #         assert jnp.all(max_diff < 1e-5), f"Causal masking test failed with max difference {max_diff}"
    #
    # def test_attention_values(self):
    #     """Test attention value computation without masking."""
    #     with self.mesh:
    #         batch_size, seq_len, num_heads, head_dim = 32, 4, 4, 32
    #
    #         # Create query that strongly attends to first position
    #         query = jnp.zeros((batch_size, seq_len, num_heads, head_dim))
    #         query = query.at[:, 1:3].set(1.0)  # Middle tokens attend strongly
    #
    #         # Create key with distinct value at first position
    #         key = jnp.zeros((batch_size, seq_len, num_heads, head_dim))
    #         key = key.at[:, 0].set(1.0)  # First position is special
    #
    #         # Create value with 1.0 at first position
    #         value = jnp.zeros((batch_size, seq_len, num_heads, head_dim))
    #         value = value.at[:, 0].set(1.0)
    #
    #         # Scale query by 1/sqrt(head_dim)
    #         query = query / jnp.sqrt(head_dim)
    #
    #         @jax.jit
    #         def flash_attention_jit(query, key, value):
    #             return flash_attention(
    #                 query=query,
    #                 key=key,
    #                 value=value,
    #                 causal=False,  # Test attention without masking
    #                 q_chunk_size=2,
    #                 kv_chunk_size=2
    #             )
    #
    #         output = flash_attention_jit(query, key, value)
    #
    #         # Middle tokens should attend strongly to first position
    #         # Other tokens should have near-zero attention
    #         expected = jnp.zeros_like(output)
    #         expected = expected.at[:, 1:3].set(1.0)
    #
    #         diff = jnp.abs(output - expected)
    #         max_diff = jnp.max(diff)
    #         assert jnp.all(max_diff < 1e-5), f"Attention values test failed with max difference {max_diff}"
    #

if __name__ == '__main__':
    absltest.main() 
