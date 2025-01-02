import jax
import jax.numpy as jnp
import optax
import logging

def combine_params_test(base_params, lora_params):
    """Simplified version of our combine_params logic"""
    combined = {}
    # First add all base params
    for k, v in base_params.items():
        combined[k] = v
    # Then overlay LoRA params
    for k, v in lora_params.items():
        if k in combined:
            combined[k] = combine_params_test(combined[k], v)
        else:
            combined[k] = v
    return combined

def main():
    # Create test parameters with same structure as our model
    base_params = {
        'attention': {
            'kernel': jnp.ones((4, 4))
        }
    }
    
    lora_params = {
        'attention': {
            'lora_A': jnp.ones((4, 2)),
            'lora_B': jnp.ones((2, 4))
        }
    }

    # Test function that combines params and does a simple operation
    def test_fn(lora_p):
        combined = combine_params_test(base_params, lora_p)
        # Simple operation that should produce non-zero gradients
        lora_a = combined['attention']['lora_A'] 
        lora_b = combined['attention']['lora_B']
        return jnp.sum(lora_a @ lora_b)

    # Check gradients
    grad_fn = jax.grad(test_fn)
    grads = grad_fn(lora_params)
    
    print("\nInput parameters:")
    print("Base params:", jax.tree_util.tree_map(lambda x: x.shape, base_params))
    print("LoRA params:", jax.tree_util.tree_map(lambda x: x.shape, lora_params))
    
    print("\nGradients:")
    print(jax.tree_util.tree_map(lambda x: (x.shape, jnp.max(jnp.abs(x))), grads))

if __name__ == "__main__":
    main()
