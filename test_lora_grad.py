import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU

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
    print("JAX devices:", jax.devices())
    
    # Create test parameters with same structure as our model
    base_params = {
        'params': {
            'attention': {
                'kernel': jnp.ones((4, 4))
            }
        }
    }
    
    lora_params = {
        'params': {
            'attention': {
                'lora_A': jnp.ones((4, 2)),
                'lora_B': jnp.ones((2, 4))
            }
        }
    }

    print("\nTest function evaluation:")
    def loss_and_accuracy(lora_p):
        # Match real training - combine params then wrap in dict
        combined = {'params': combine_params_test(base_params, lora_p)}
        print("Combined params:", jax.tree_util.tree_map(lambda x: x.shape, combined))
        # Simple operation that should produce non-zero gradients
        lora_a = combined['params']['params']['attention']['lora_A'] 
        lora_b = combined['params']['params']['attention']['lora_B']
        return jnp.sum(lora_a @ lora_b), 0.0  # Return dummy accuracy like real code

    # First run function normally
    result, _ = jax.jit(loss_and_accuracy)(lora_params)
    print("Test function result:", float(result))  # Safe to convert after jit

    # Then check gradients - match real training grad_fn setup
    grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
    (loss, _), grads = jax.jit(grad_fn)(lora_params)

    # First run function normally
    result = jax.jit(test_fn)(lora_params)
    print("Test function result:", float(result))  # Safe to convert after jit

    # Then check gradients
    print("\nGradient computation:")
    grad_fn = jax.grad(test_fn)
    grads = jax.jit(grad_fn)(lora_params)  # Jit the gradient computation
    
    print("\nInput parameters:")
    print("Base params:", jax.tree_util.tree_map(lambda x: x.shape, base_params))
    print("LoRA params:", jax.tree_util.tree_map(lambda x: x.shape, lora_params))
    
    print("\nGradients:")
    # Convert to concrete values after computation
    grads_info = jax.tree_util.tree_map(
        lambda x: (x.shape, float(jax.device_get(jnp.max(jnp.abs(x))))), 
        grads
    )
    print(grads_info)

if __name__ == "__main__":
    main()
