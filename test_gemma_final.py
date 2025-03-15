#!/usr/bin/env python
# coding=utf-8

import torch
import logging
import os
import json
import time

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def test_gemma_inference():
    """
    Comprehensive approach to load Gemma-3 model addressing all nested attributes.
    """
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN environment variable not set. You may encounter authentication issues.")
    
    logger.info(f"Found {torch.cuda.device_count()} CUDA devices")
    
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # 1. Create a custom Gemma3Config class that addresses the nested structure
        from transformers import Gemma3Config as OriginalGemma3Config
        from transformers import AutoTokenizer
        
        class CustomGemma3Config(OriginalGemma3Config):
            """Custom config that automatically copies all text_config attributes to root level."""
            
            @classmethod
            def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
                config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
                
                # Get tokenizer to extract vocab_size if needed
                try:
                    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
                    if hasattr(tokenizer, 'vocab_size'):
                        logger.info(f"Got vocab_size from tokenizer: {tokenizer.vocab_size}")
                        config.vocab_size = tokenizer.vocab_size
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer: {e}")
                
                # Extract and copy all text_config attributes
                config_dict = config.to_dict()
                if 'text_config' in config_dict and isinstance(config_dict['text_config'], dict):
                    for key, value in config_dict['text_config'].items():
                        logger.info(f"Copying attribute from text_config: {key}={value}")
                        setattr(config, key, value)
                
                # Manually set sliding_window_pattern if needed
                if hasattr(config, "sliding_window") and not hasattr(config, "sliding_window_pattern"):
                    # Default pattern that seems common in Gemma models
                    setattr(config, "sliding_window_pattern", 6)
                    logger.info(f"Manually added sliding_window_pattern=6 based on common patterns")
                
                return config
        
        model_name = "google/gemma-3-27b-pt"
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
        )
        logger.info(f"Tokenizer loaded with vocab_size: {tokenizer.vocab_size}")
        
        # Load custom config that auto-copies all text_config attributes
        logger.info("Loading and patching config with custom class")
        config = CustomGemma3Config.from_pretrained(model_name, token=hf_token)
        
        # Log all available config attributes for debugging
        config_attrs = [attr for attr in dir(config) if not attr.startswith('_') and not callable(getattr(config, attr))]
        logger.info(f"Config now has attributes: {config_attrs}")
        
        # Double-check critical attributes
        critical_attrs = [
            "vocab_size", "hidden_size", "intermediate_size", "num_attention_heads", 
            "num_hidden_layers", "sliding_window", "sliding_window_pattern"
        ]
        for attr in critical_attrs:
            if hasattr(config, attr):
                logger.info(f"✅ Config has {attr}: {getattr(config, attr)}")
            else:
                logger.warning(f"❌ Config missing critical attribute: {attr}")
        
        # Get the correct model class
        from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
        
        # Load model with fully patched config
        logger.info("Loading model with fully patched config...")
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token=hf_token,
            # Try loading without device_map first to avoid meta device issues
            # We'll move it to the appropriate device after loading
        )
        
        # After successful loading, check embedding layer
        logger.info("Checking embedding layer")
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            if hasattr(model.model.embed_tokens, 'weight'):
                embed_shape = model.model.embed_tokens.weight.shape
                embed_device = model.model.embed_tokens.weight.device
                logger.info(f"Embed tokens weight shape: {embed_shape}")
                logger.info(f"Embed tokens device: {embed_device}")
            else:
                logger.info("Embed tokens has no weight attribute")
        
        # Test inference
        test_input = "Hello, my name is"
        logger.info(f"Running inference with input: '{test_input}'")
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="pt")
        
        # Move model to appropriate device if not already there
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Moving model to {device}")
            model = model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        logger.info("Starting generation...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=20, 
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode and print output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Model output: '{generated_text}'")
        logger.info("✅ Inference successful!")
        
        return True
        
    except Exception as e:
        logger.error(f"Final approach failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    start_time = time.time()
    success = test_gemma_inference()
    end_time = time.time()
    
    if success:
        logger.info(f"✅ Test completed successfully in {end_time - start_time:.2f} seconds")
    else:
        logger.error(f"❌ Test failed after {end_time - start_time:.2f} seconds") 