#!/usr/bin/env python
# coding=utf-8

import torch
import logging
import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def test_gemma_inference():
    """
    Simple test to check if the base Gemma-3 model works for inference without LoRA.
    Directly patching config to fix vocab_size issue.
    """
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN environment variable not set. You may encounter authentication issues.")
    
    model_name = "google/gemma-3-27b-pt"  # 27B parameter model
    
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        token=hf_token,
    )
    
    logger.info(f"Tokenizer loaded with vocab_size: {tokenizer.vocab_size}")
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and fix the config first
    logger.info(f"Loading and patching config from {model_name}")
    config = AutoConfig.from_pretrained(model_name, token=hf_token)
    
    # Monkey patch the config object directly
    # Add vocab_size attribute directly to the config object, not just its dict
    if not hasattr(config, "vocab_size"):
        logger.info("Patching vocab_size directly onto config object")
        vocab_size = getattr(tokenizer, "vocab_size", 262144)
        setattr(config, "vocab_size", vocab_size)
        logger.info(f"Set vocab_size to {vocab_size}")
    
    # Manually copy text_config attributes to main config if needed
    if hasattr(config, "text_config") and isinstance(config.text_config, dict):
        for key, value in config.text_config.items():
            if not hasattr(config, key):
                logger.info(f"Copying {key} from text_config to main config")
                setattr(config, key, value)
    
    # Print config attributes for debugging
    config_attrs = [attr for attr in dir(config) if not attr.startswith('_') and not callable(getattr(config, attr))]
    logger.info(f"Config now has attributes: {config_attrs}")
    
    logger.info(f"Loading model from {model_name}")
    logger.info(f"Found {torch.cuda.device_count()} CUDA devices")
    
    # Configure model loading  
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "token": hf_token,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    
    # Load model with patched config
    logger.info("Loading model with patched config...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,  # Use our patched config
        **model_kwargs
    )
    
    # Check embedding layer
    logger.info("Checking embedding layer:")
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'weight'):
        embed_shape = model.model.embed_tokens.weight.shape
        embed_device = model.model.embed_tokens.weight.device
        logger.info(f"Embed tokens weight shape: {embed_shape}")
        logger.info(f"Embed tokens device: {embed_device}")
    elif hasattr(model, 'get_input_embeddings'):
        embed = model.get_input_embeddings()
        if embed is not None and hasattr(embed, 'weight'):
            logger.info(f"Input embeddings weight shape: {embed.weight.shape}")
            logger.info(f"Input embeddings device: {embed.weight.device}")
        else:
            logger.info("Input embeddings not found or has no weight attribute")
    else:
        logger.info("Could not find embeddings through standard paths")
    
    # Test inference
    test_input = "Hello, my name is"
    logger.info(f"Running inference with input: '{test_input}'")
    
    # Tokenize input
    input_ids = tokenizer(test_input, return_tensors="pt")
    
    # Move to appropriate device
    input_device = next(model.parameters()).device
    logger.info(f"Moving input tensors to {input_device}")
    input_ids = {k: v.to(input_device) for k, v in input_ids.items()}
    
    # Generate
    with torch.no_grad():
        try:
            logger.info("Starting generation...")
            outputs = model.generate(
                **input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
            )
            
            # Decode and print output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Model output: '{generated_text}'")
            logger.info("✅ Inference successful!")
            
        except Exception as e:
            logger.error(f"❌ Inference failed with error: {str(e)}")
            logger.error("This suggests an issue with the base model itself.")
            
            # Try to get more information about the error
            import traceback
            logger.error(traceback.format_exc())
    
    # Log memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"GPU {i} memory allocated: {memory_allocated:.2f} GB")
            logger.info(f"GPU {i} memory reserved: {memory_reserved:.2f} GB")

if __name__ == "__main__":
    test_gemma_inference() 