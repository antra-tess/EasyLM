import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as PS
from absl.testing import absltest, parameterized
from EasyLM.jax_utils import with_sharding_constraint, get_jax_mesh

from EasyLM.attention.flash_attention import flash_attention

class FlashAttentionTest(parameterized.TestCase):
    def setUp(self):
        # Set up mesh using LLaMAConfigurator's method
        self.mesh = get_jax_mesh((2, -1, 2), ("dp", "fsdp", "mp"))
        
    def get_attention_inputs(self, batch=1, seq_len=16, num_q_heads=8, num_kv_heads=4, head_dim=32):
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
            
            # Compare
            np.testing.assert_allclose(
                flash_output, ref_output, 
                rtol=1e-5, atol=1e-5
            )

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
