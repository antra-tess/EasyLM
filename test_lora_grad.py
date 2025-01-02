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
    # Base case: if either input is not a dict, return base params
    if not isinstance(base_params, dict) or not isinstance(lora_params, dict):
        return base_params

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
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        # Convert shapes to concrete values to avoid weak typing
        input_dim = int(x.shape[-1])
        output_dim = int(self.features)
        rank = int(self.lora_rank)
        
        x = x.astype(jnp.float32)  # Ensure input is float32
        
        # Base weight with concrete shapes
        kernel = self.param('kernel', 
            jax.nn.initializers.normal(0.02),
            (input_dim, output_dim),
            jnp.float32)
            
        # LoRA weights with concrete shapes
        lora_alpha = 8.0  # Like in real LoRA
        scaling = lora_alpha / rank
        
        lora_A = self.param('lora_A',
            jax.nn.initializers.normal(0.02),
            (input_dim, rank),
            jnp.float32)
        lora_B = self.param('lora_B', 
            jax.nn.initializers.normal(0.02),
            (rank, output_dim),
            jnp.float32)
            
        # Compute output with LoRA
        base_out = jnp.matmul(x, kernel)
        intermediate = jnp.matmul(x, lora_A)
        delta = jnp.matmul(intermediate, lora_B) * scaling
        
        # Debug prints for key values only
        jax.debug.print("intermediate max: {}", jnp.max(jnp.abs(intermediate)))
        jax.debug.print("delta max: {}", jnp.max(jnp.abs(delta)))
        jax.debug.print("base_out max: {}", jnp.max(jnp.abs(base_out)))
        
        return base_out + delta

class SimpleModel(nn.Module):
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        x = SimpleLoRALinear(features=8, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = jax.nn.relu(x)
        x = SimpleLoRALinear(features=8, dtype=self.dtype, param_dtype=self.param_dtype)(x)
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
    
    # Create constant input and target as float32
    input_data = jnp.ones((1, 16), dtype=jnp.float32)  # Single constant input
    target = jnp.ones((1, 8), dtype=jnp.float32)  # Single constant target
    
    # Initialize model and parameters
    model = SimpleModel()
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    # Initialize all parameters first
    params = model.init(init_rng, input_data)
    
    # Split into base and LoRA parameters
    flat_params = flatten_dict(params)
    base_dict = {}
    lora_dict = {}
    
    for k, v in flat_params.items():
        if k[-1] == 'kernel':  # Base parameters
            base_dict[k] = v.astype(jnp.float32)
        elif k[-1] in ['lora_A', 'lora_B']:  # LoRA parameters
            lora_dict[k] = v.astype(jnp.float32)
    
    base_params = {'params': unflatten_dict(base_dict)}
    lora_params = {'params': unflatten_dict(lora_dict)}
    
    # Create optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(lora_params)
    
    # Training loop
    train_state = {'params': lora_params, 'opt_state': opt_state}
    
    def loss_fn(lora_params):
        # Extract just the LoRA parameters we want to differentiate
        lora_dict = lora_params['params']
        
        # Combine with base parameters for forward pass
        combined = combine_params_test(base_params['params'], lora_dict)
        
        # Run forward pass and compute loss
        output = model.apply({'params': combined['params']}, input_data)
        diff = output - target
        loss = jnp.mean(diff * diff)
        
        return loss

    # Training loop
    train_state = {'params': lora_params, 'opt_state': opt_state}
    
    for step in range(2):
        # Compute loss and gradients with respect to LoRA params only
        loss_val, grads = jax.value_and_grad(loss_fn)(train_state['params'])
        jax.debug.print("Raw gradients: {}", 
                       jax.tree_util.tree_map(lambda x: jnp.max(jnp.abs(x)), grads))
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, train_state['opt_state'])
        new_params = optax.apply_updates(train_state['params'], updates)
        train_state = {'params': new_params, 'opt_state': new_opt_state}
        
        # Print metrics
        grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads)))
        print(f"\nStep {step}:")
        print(f"Loss: {loss_val}")
        print(f"Gradient norm: {grad_norm}")

    for step in range(2):
        # Compute loss and gradients
        loss_val, grads = jax.value_and_grad(loss_fn)(train_state['params'])
        jax.debug.print("Raw gradients: {}", 
                       jax.tree_util.tree_map(lambda x: jnp.max(jnp.abs(x)), grads))
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, train_state['opt_state'])
        new_params = optax.apply_updates(train_state['params'], updates)
        train_state = {'params': new_params, 'opt_state': new_opt_state}
        
        # Print metrics
        grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads)))
        print(f"\nStep {step}:")
        print(f"Loss: {loss_val}")
        print(f"Gradient norm: {grad_norm}")

if __name__ == "__main__":
    main()
