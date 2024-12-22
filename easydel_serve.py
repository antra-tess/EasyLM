import os
import jax
import easydel as ed
import jax.numpy as jnp
from transformers import AutoTokenizer
print("After imports")

# Configuration
model_path = "/dev/shm"  # Path to our converted model
hf_model_name = "meta-llama/Meta-Llama-3-8B"  # For tokenizer
max_length = 8192
print("Before first jax call")
num_devices = jax.device_count()
print("After first jax call")

print(f"Loading tokenizer from {hf_model_name}...")
tokenizer = AutoTokenizer.from_pretrained(
    hf_model_name,
    use_auth_token=True,  # Uses HF_TOKEN from environment
    trust_remote_code=True
)

print("Loading model...")
model, params = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_path,
    sharding_axis_dims=(1, 1, 1, -1),  # Single worker for now
    auto_shard_params=True,
    dtype=jnp.float16,
    param_dtype=jnp.float16,
    precision=None,
    input_shape=(num_devices, max_length),
    quantization_method=ed.EasyDeLQuantizationMethods.A8BIT,
    config_kwargs=ed.EasyDeLBaseConfigDict(
        quantize_kv_cache=True,
        attn_dtype=jnp.float16,
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        mask_max_position_embeddings=max_length,
        freq_max_position_embeddings=max_length,
    ),
)

print("Initializing inference...")
inference = ed.vInference(
    model=model,
    params=params,
    tokenizer=tokenizer,
    generation_config=ed.vInferenceConfig(
        temperature=0.8,
        top_k=10,
        top_p=0.95,
        streaming_chunks=32,
        max_new_tokens=1024,
    ),
)

print("Precompiling model...")
inference.precompile(batch_size=1)

print("Running test inference...")
test_prompt = "Write a short poem about a cat."
print(f"\nPrompt: {test_prompt}")
print("\nGenerating response...")
response = inference.generate_text(test_prompt)
print(f"\nResponse:\n{response}")
