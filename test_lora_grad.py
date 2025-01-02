import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as PS
from jax.experimental.pjit import pjit
import optax
import logging

def get_jax_mesh(mesh_shape, axis_names):
    device_ids = jnp.array(range(jax.device_count())).reshape(mesh_shape)
    mesh = Mesh(device_ids, axis_names)
    return mesh

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

    # Move entire parameter trees to device
    base_params = jax.device_put(base_params)
    lora_params = jax.device_put(lora_params)

    # Define partition specs
    base_param_partition = {
        'params': {
            'params': {
                'attention': {
                    'kernel': PS('mp', 'fsdp'),
                    'lora_A': PS('fsdp', None),
                    'lora_B': PS(None, 'mp')
                }
            }
        }
    }
    
    lora_param_partition = {
        'params': {
            'params': {
                'attention': {
                    'lora_A': PS('fsdp', None),
                    'lora_B': PS(None, 'mp')
                }
            }
        }
    }

    print("\nTest function evaluation:")
    print("\nTesting approach 1 - differentiate wrt all parameters:")
    def loss_fn_all(all_params):
        # Simple operation that should produce non-zero gradients
        lora_a = all_params['params']['params']['attention']['lora_A']
        lora_b = all_params['params']['params']['attention']['lora_B']
        return jnp.sum(lora_a @ lora_b), 0.0

    # Use regular jit since we're testing on CPU
    print("\nTesting approach 1 - differentiate wrt all parameters:")
    jitted_loss_fn_all = jax.jit(loss_fn_all)
    grad_fn_all = jax.value_and_grad(loss_fn_all, has_aux=True)
    jitted_grad_fn_all = jax.jit(grad_fn_all)

    print("\nTesting approach 2 - differentiate wrt LoRA parameters only:")
    def loss_fn_lora(lora_p):
        # Combine with base params for forward pass
        combined = {'params': combine_params_test(base_params, lora_p)}
        # Simple operation that should produce non-zero gradients
        lora_a = combined['params']['params']['attention']['lora_A']
        lora_b = combined['params']['params']['attention']['lora_B']
        return jnp.sum(lora_a @ lora_b), 0.0

    jitted_loss_fn_lora = jax.jit(loss_fn_lora)
    grad_fn_lora = jax.value_and_grad(loss_fn_lora, has_aux=True)
    jitted_grad_fn_lora = jax.jit(grad_fn_lora)

    # Test approach 1
    combined_params = {'params': combine_params_test(base_params, lora_params)}
    result, _ = jitted_loss_fn_all(combined_params)
    print("Test function result (all params):", float(result))

    (loss, _), grads_all = jitted_grad_fn_all(combined_params)
    print("\nGradients when differentiating all parameters:")
    print(jax.tree_util.tree_map(lambda x: (x.shape, float(jax.device_get(jnp.max(jnp.abs(x))))), grads_all))

    # Test approach 2
    result, _ = jitted_loss_fn_lora(lora_params)
    print("Test function result (LoRA only):", float(result))

    (loss, _), grads_lora = jitted_grad_fn_lora(lora_params)
    print("\nGradients when differentiating LoRA parameters only:")
    print(jax.tree_util.tree_map(lambda x: (x.shape, float(jax.device_get(jnp.max(jnp.abs(x))))), grads_lora))

if __name__ == "__main__":
    main()
