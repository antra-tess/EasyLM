Below is one concrete way to fix the problem by **stopping gradients through the large “base” matmul outputs** whenever the base weights are frozen (that is, when LoRA is enabled and the base kernel is not being trained). If you only do `jax.lax.stop_gradient` on the base parameters, JAX will still keep references to those big intermediate outputs in the backward pass.  

By inserting a `stop_gradient` call on the result of the main base matmul, JAX no longer needs to store that partial for gradient calculations. This typically resolves the LoRA‐only OOM issue. (Of course, you can refine these changes or make them more configurable, but the example below should illustrate the gist of it.)

---

### 1. Modify `LoRALinear.__call__` to stop gradients on the base matmul output

In **`llama_model.py`**, look for your `LoRALinear` class. A typical version might look like this (simplified):

```python
class LoRALinear(nn.Module):
    features: int
    use_bias: bool = False
    config: Any = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param(
            "kernel",
            jax.nn.initializers.normal(stddev=0.02),
            (x.shape[-1], self.features),
            self.param_dtype,
        )
        bias = None
        if self.use_bias:
            bias = self.param(
                "bias",
                jax.nn.initializers.zeros,
                (self.features,),
                self.param_dtype,
            )

        # ------------------------------
        # Before: simple matmul
        # y = jnp.matmul(x, kernel.astype(self.dtype))
        #
        # Here is the new approach:
        # ------------------------------

        # Always compute the base matmul:
        base_out = jnp.matmul(x, kernel.astype(self.dtype))

        # If LoRA rank > 0, we are (likely) freezing the base kernel.
        # So we must also stop gradients through the base_out itself.
        # That ensures JAX does not build large backprop references for the base matmul.
        if self.config.lora_rank > 0:
            # We do not want to backprop into the base_out
            base_out = jax.lax.stop_gradient(base_out)

        # If LoRA is completely disabled (lora_rank=0), this is normal dense behavior.
        # If LoRA is enabled, we compute the low-rank delta and add it in.
        if self.config.lora_rank == 0:
            y = base_out
        else:
            lora_A = self.param(
                "lora_A",
                jax.nn.initializers.zeros,
                (x.shape[-1], self.config.lora_rank),
                self.param_dtype,
            )
            lora_B = self.param(
                "lora_B",
                jax.nn.initializers.zeros,
                (self.config.lora_rank, self.features),
                self.param_dtype,
            )
            scaling = self.config.lora_alpha / self.config.lora_rank

            # Possibly apply dropout to x for LoRA path
            if self.config.lora_dropout > 0.0:
                x_lora = nn.Dropout(rate=self.config.lora_dropout)(
                    x, deterministic=not self.is_mutable_collection("dropout")
                )
            else:
                x_lora = x

            # Delta is the trainable part
            delta = jnp.matmul(x_lora, lora_A.astype(self.dtype))
            delta = jnp.matmul(delta, lora_B.astype(self.dtype)) * scaling

            # Final output is base_out + delta
            y = base_out + delta

        if bias is not None:
            y = y + bias.astype(self.dtype)

        return y
```

**Explanation**:  
- We used `base_out = jax.lax.stop_gradient(base_out)` whenever `lora_rank > 0`. That means if you intend to freeze the base kernel (typical for LoRA), you also sever the gradient tape in the matmul’s output. This is what prevents the large activation memory usage in the backward pass.  
- If `lora_rank = 0`, it means a normal training scenario (no LoRA), so we just do the usual dense matmul with a gradient path.  

If you have a **more granular** approach—such as only freezing some layers or only attention weights but not MLP—then you would need to conditionally apply `stop_gradient` only when the base matmul is truly frozen. In that case, you might rely on a per‐layer or per‐parameter “trainable” flag. The key idea stays the same: if the base param is frozen, also `stop_gradient` on that base matmul output.

---

### 2. Remove or simplify the “stop_grad on the param” logic in your training step

You already have a snippet that does:

```python
def maybe_stop_grad(path, p):
    if not trainable_mask(path):
        return jax.lax.stop_gradient(p)
    return p
```

and then you feed `params_for_forward` into the model. While that is fine for ensuring “no gradient w.r.t. the base param,” you **also** need the code snippet above to avoid big memory usage. Because if you only do `stop_gradient(p)`, JAX will still backprop through the `x @ kernel` operation to find \(\nabla x\). So you can keep your existing logic if you like, but the crucial fix is the `stop_gradient(base_out)` in `LoRALinear`.

---

### 3. Double‐check training logs

If done correctly, the memory usage for your LoRA‐only run should drop dramatically. You should see fewer or no OOM errors (depending on your hardware constraints).  

Also, do a short run to confirm that the gradient w.r.t. LoRA parameters is nonzero (the point of LoRA training!). You can do that by printing or logging the norm of your “lora_A” and “lora_B” parameter updates. You should see them updating as expected.

---

### Final note on memory

This approach is the simplest single fix that typically solves the LoRA OOM problem. The same idea can be applied to other spots if needed (e.g. attention projections). But **most** of the time, simply intercepting these big base matmuls in `LoRALinear` and calling `stop_gradient` there is enough to unburden the memory usage.
