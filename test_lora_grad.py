import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import Mesh, PartitionSpec as PS
from jax.experimental.pjit import pjit
import optax
from typing import Any
import logging

def get_mesh():
    """Create simple mesh for testing."""
    mesh = Mesh(np.array(jax.devices()), ('dp',))  # Single axis mesh for testing
    return mesh

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
    
    @nn.compact
    def __call__(self, x):
        # Base weight
        kernel = self.param('kernel', 
            jax.nn.initializers.normal(0.02),
            (x.shape[-1], self.features))
            
        # LoRA weights with scaling
        lora_alpha = 8.0  # Like in real LoRA
        scaling = lora_alpha / self.lora_rank
        
        lora_A = self.param('lora_A',
            jax.nn.initializers.normal(0.02),  # Small non-zero init
            (x.shape[-1], self.lora_rank))
        lora_B = self.param('lora_B', 
            jax.nn.initializers.normal(0.02),  # Small non-zero init
            (self.lora_rank, self.features))
            
        # Compute output with LoRA
        base_out = jnp.matmul(x, kernel)
        intermediate = jnp.matmul(x, lora_A)
        delta = jnp.matmul(intermediate, lora_B) * scaling
        
        # Debug prints
        jax.debug.print("x shape: {}", x.shape)
        jax.debug.print("kernel shape: {}", kernel.shape)
        jax.debug.print("lora_A shape: {}", lora_A.shape)
        jax.debug.print("lora_B shape: {}", lora_B.shape)
        jax.debug.print("intermediate shape: {}", intermediate.shape)
        jax.debug.print("intermediate max: {}", jnp.max(jnp.abs(intermediate)))
        jax.debug.print("delta shape: {}", delta.shape)
        jax.debug.print("delta max: {}", jnp.max(jnp.abs(delta)))
        jax.debug.print("base_out shape: {}", base_out.shape)
        jax.debug.print("base_out max: {}", jnp.max(jnp.abs(base_out)))
        
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
    
    # Set up pjit
    param_partition = PS(None)  # No sharding for testing
    
    def train_step(train_state, base_params, batch):
        def loss_fn(params):
            output = model.apply(params, batch)
            return jnp.mean((output - target) ** 2)

        # Combine parameters before gradient computation
        combined_params = {'params': combine_params_test(base_params['params'], train_state['params'])}
        
        # Compute loss and gradients
        loss_val, grads = jax.value_and_grad(loss_fn)(combined_params)
        jax.debug.print("Raw gradients: {}", 
                       jax.tree_util.tree_map(lambda x: jnp.max(jnp.abs(x)), grads))
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, train_state['opt_state'])
        new_params = optax.apply_updates(train_state['params'], updates)
        
        metrics = {
            'loss': loss_val,
            'grad_norm': jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads)))
        }
        
        return {'params': new_params, 'opt_state': new_opt_state}, metrics

    # Wrap with pjit
    sharded_train_step = pjit(
        train_step,
        in_shardings=(param_partition, param_partition, param_partition),
        out_shardings=(param_partition, param_partition)
    )

    # Training loop with mesh context
    mesh = get_mesh()
    with mesh:
        train_state = {'params': lora_params, 'opt_state': opt_state}
        
        for step in range(3):
            train_state, metrics = sharded_train_step(
                train_state, base_params, input_data
            )
            
            # Print metrics
            print(f"\nStep {step}:")
            print(f"Loss: {metrics['loss']}")
            print(f"Gradient norm: {metrics['grad_norm']}")

if __name__ == "__main__":
    main()
