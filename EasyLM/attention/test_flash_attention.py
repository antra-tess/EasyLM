import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as PS
from absl.testing import absltest, parameterized
from EasyLM.jax_utils import with_sharding_constraint, get_jax_mesh
from jax.experimental.multihost_utils import process_allgather

from EasyLM.attention.flash_attention import flash_attention


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
            query = with_sharding_constraint(query, PS(("dp",), None, "mp", None))
            key = with_sharding_constraint(key_tensor, PS(("dp",), None, "mp", None))
            value = with_sharding_constraint(value, PS(("dp",), None, "mp", None))

        return query, key, value

    def reference_attention(self, query, key, value, causal=True):
        """Reference implementation for comparison."""
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

        # Apply softmax
        scores = jax.nn.softmax(scores, axis=-1)

        # Compute output
        output = jnp.einsum('bhqk,bkhd->bqhd', scores, value)
        return output

    @parameterized.parameters(
        {"seq_len": 16, "num_q_heads": 8, "num_kv_heads": 4},
        {"seq_len": 32, "num_q_heads": 16, "num_kv_heads": 4},
    )
    def test_flash_attention_matches_reference(self, seq_len, num_q_heads, num_kv_heads):
        """Test that flash attention matches reference implementation."""
        with self.mesh:
            query, key, value = self.get_attention_inputs(
                seq_len=seq_len,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads
            )

            # Run flash attention
            flash_output = flash_attention(
                query=query,
                key=key,
                value=value,
                chunk_size=8  # Small chunk size for testing
            )

            # Run reference
            ref_output = self.reference_attention(query, key, value)

            # Gather results from all devices for comparison
            from jax.experimental.multihost_utils import process_allgather

            flash_gathered = process_allgather(flash_output)
            ref_gathered = process_allgather(ref_output)

            if jax.process_index() == 0:  # Only compare on main process
                np.testing.assert_allclose(
                    flash_gathered, ref_gathered,
                    rtol=1e-5, atol=1e-5
                )

    @parameterized.parameters(
        {"causal": True},
        {"causal": False}
    )
    def test_causal_masking(self, causal):
        """Test that causal masking properly blocks future tokens."""
        with self.mesh:
            # Create inputs where later tokens should strongly attend to earlier ones
            batch_size, seq_len, num_heads, head_dim = 32, 4, 4, 32  # Match mesh dimensions: fsdp=8, mp=4
            query = jnp.ones((batch_size, seq_len, num_heads, head_dim))
            key = value = jnp.ones((batch_size, seq_len, num_heads, head_dim))

            # Make later tokens have strong attention to earlier ones
            value = value * jnp.arange(1, 5).reshape(1, 4, 1, 1)

            # Run attention with and without causal mask
            output = flash_attention(
                query=query,
                key=key,
                value=value,
                causal=causal,
                chunk_size=2
            )

            if causal:
                # Each token should only see values up to its position
                expected = jnp.broadcast_to(
                    jnp.array([[[[1.]], [[1.5]], [[2.]], [[2.5]]]]),
                    (batch_size, seq_len, num_heads, head_dim)
                )
            else:
                # All tokens should see all values (average = 2.5)
                expected = jnp.ones((batch_size, seq_len, num_heads, head_dim)) * 2.5

            diff = jnp.abs(output - expected)
            max_diff = jnp.max(diff)
            assert jnp.all(max_diff < 1e-5), f"Causal masking test failed with max difference {max_diff}"

    def test_attention_patterns(self):
        """Test specific attention patterns."""
        with self.mesh:
            # Create a sequence where middle tokens should attend strongly to first token
            batch_size, seq_len, num_heads, head_dim = 32, 4, 4, 32  # Match mesh dimensions: fsdp=8, mp=4
            query = jnp.zeros((batch_size, seq_len, num_heads, head_dim))
            key = value = jnp.zeros((batch_size, seq_len, num_heads, head_dim))

            # Set up pattern: first token has value 1, others 0
            value = value.at[:, 0, :, :].set(1.0)
            # Middle tokens attend only to first token
            query = query.at[:, 1:3, :, :].set(10.0)  # Strong attention
            key = key.at[:, 0, :, :].set(10.0)  # Strong key for first token

            output = flash_attention(
                query=query,
                key=key,
                value=value,
                causal=False,
                chunk_size=2
            )

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

    def test_invalid_head_config(self):
        """Test that invalid head configurations raise error."""
        with self.assertRaises(ValueError):
            # num_q_heads not divisible by num_kv_heads
            query, key, value = self.get_attention_inputs(
                num_q_heads=6,  # Not divisible by 4
                num_kv_heads=4
            )
            flash_attention(query, key, value)


if __name__ == '__main__':
    absltest.main() 
