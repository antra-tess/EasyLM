#!/usr/bin/env python
# coding=utf-8

import torch
import logging
import os
import json
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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
    Optimized for running on 2xA100 GPUs.
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
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and fix the config first
    logger.info(f"Loading and preprocessing config from {model_name}")
    config = AutoConfig.from_pretrained(model_name, token=hf_token)
    
    # Check if config has the nested structure and fix it
    config_dict = config.to_dict()
    
    if 'text_config' in config_dict:
        logger.info("Found nested text_config structure, unfolding parameters...")
        text_config = config_dict.get('text_config', {})
        
        # For debugging, let's print what's inside text_config
        logger.info(f"Text config contains keys: {list(text_config.keys())}")
        
        # Set vocab_size explicitly if not present
        if 'vocab_size' not in text_config and hasattr(tokenizer, 'vocab_size'):
            text_config['vocab_size'] = tokenizer.vocab_size
            logger.info(f"Setting vocab_size from tokenizer: {tokenizer.vocab_size}")
        elif 'vocab_size' not in text_config:
            # Gemma-3 uses 262144 tokens
            text_config['vocab_size'] = 262144
            logger.info("Setting default vocab_size to 262144")
        
        # Create a new config with unfolded parameters
        for key, value in text_config.items():
            if key not in config_dict:
                config_dict[key] = value
                logger.info(f"Unfolded parameter {key} to root level")
        
        # Create a new config object
        try:
            from transformers import Gemma3Config
            new_config = Gemma3Config.from_dict(config_dict)
            logger.info("Successfully created new config with unfolded parameters")
            config = new_config
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not create Gemma3Config: {e}. Using modified dict instead.")
            # Just modify the original config directly
            for key, value in text_config.items():
                setattr(config, key, value)
    
    logger.info(f"Loading model from {model_name}")
    
    # Configure model loading
    compute_dtype = torch.bfloat16  # Use bfloat16 for more efficient inference
    
    # Create a temp directory for potential weight offloading
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory for weight offloading: {temp_dir}")
    
    # Set up model loading parameters with adjusted settings
    model_kwargs = {
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
        "token": hf_token,
        "device_map": "auto",  # Let transformers handle multi-GPU placement
        "max_memory": {i: "80GiB" for i in range(torch.cuda.device_count())},  # Allocate memory for each GPU
        "config": config,  # Use our modified config
        "low_cpu_mem_usage": True,  # Enable low CPU memory usage
        "offload_folder": temp_dir,  # Specify offload folder
        "offload_state_dict": True,  # Enable state dict offloading
    }
    
    logger.info(f"Loading with kwargs: {model_kwargs}")
    logger.info(f"Found {torch.cuda.device_count()} CUDA devices")
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Check and log embedding layer status
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
    
    # Test with a simple input
    test_input = "Hello, my name is"
    
    logger.info(f"Running inference with input: '{test_input}'")
    
    # Tokenize input
    input_ids = tokenizer(test_input, return_tensors="pt")
    
    # For device_map="auto", transformers will handle the placement automatically
    # We just need to ensure input tensors are on the right device
    input_device = next(model.parameters()).device
    logger.info(f"Moving input tensors to {input_device}")
    input_ids = {k: v.to(input_device) for k, v in input_ids.items()}
    
    # Generate output
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
                use_cache=True,  # Can use use_cache here in generate() method
            )
            
            # Decode and print output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Model output: '{generated_text}'")
            logger.info("✅ Inference successful!")
            
            # Check embedding shape after inference
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'weight'):
                embed_shape = model.model.embed_tokens.weight.shape
                logger.info(f"Embed tokens weight shape after inference: {embed_shape}")
            
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
    
    # Clean up temp directory
    import shutil
    try:
        shutil.rmtree(temp_dir)
        logger.info(f"Removed temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Could not remove temporary directory {temp_dir}: {e}")

if __name__ == "__main__":
    test_gemma_inference() 