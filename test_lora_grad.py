import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.traverse_util import flatten_dict, unflatten_dict
import optax
from typing import Any
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

class SimpleLoRALinear(nn.Module):
    features: int
    lora_rank: int = 8
    
    @nn.compact
    def __call__(self, x):
        # Base weight
        kernel = self.param('kernel', 
            jax.nn.initializers.normal(0.02),
            (x.shape[-1], self.features))
            
        # LoRA weights
        lora_A = self.param('lora_A',
            jax.nn.initializers.zeros,
            (x.shape[-1], self.lora_rank))
        lora_B = self.param('lora_B', 
            jax.nn.initializers.zeros,
            (self.lora_rank, self.features))
            
        # Compute output with LoRA
        base_out = jnp.matmul(x, kernel)
        delta = jnp.matmul(jnp.matmul(x, lora_A), lora_B)
        return base_out + delta

class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = SimpleLoRALinear(features=8)(x)
        x = jax.nn.relu(x)
        x = SimpleLoRALinear(features=8)(x)
        return x

def create_train_state(rng, model, input_shape, learning_rate=1e-3):
    """Creates initial `TrainState` for model."""
    params = model.init(rng, jnp.ones(input_shape))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

def main():
    print("JAX devices:", jax.devices())
    
    # Create constant input and target
    input_data = jnp.ones((1, 16))  # Single constant input
    target = jnp.ones((1, 8))  # Single constant target
    
    # Initialize model and parameters
    model = SimpleModel()
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    # Initialize base parameters with random values
    base_params = model.init(init_rng, input_data)
    
    # Initialize LoRA parameters with zeros
    flat_params = flatten_dict(base_params)
    lora_params = {}
    for k, v in flat_params.items():
        path_str = '/'.join(str(x) for x in k)
        if 'lora' in path_str:
            lora_params[k] = jnp.zeros_like(v)
        else:
            lora_params[k] = v
    lora_params = unflatten_dict(lora_params)
    
    # Create optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(lora_params)
    
    # Training loop
    for step in range(1000):
        def loss_fn(params):
            # Combine parameters for forward pass
            combined = {'params': combine_params_test(base_params, params)}
            output = model.apply(combined, input_data)
            return jnp.mean((output - target) ** 2)
        
        # Compute loss and gradients
        loss_val, grads = jax.value_and_grad(loss_fn)(lora_params)
        
        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss_val}")
            print("Gradient norms:")
            flat_grads = flatten_dict(grads)
            for k, v in flat_grads.items():
                if 'lora' in str(k):
                    print(f"  {k}: {jnp.linalg.norm(v)}")
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        lora_params = optax.apply_updates(lora_params, updates)

if __name__ == "__main__":
    main()
