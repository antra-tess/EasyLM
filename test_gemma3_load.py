#!/usr/bin/env python
# coding=utf-8

import torch
import logging
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    """
    Simple test script to verify Gemma-3 model loading works correctly
    """
    model_name = "google/gemma-3-27b-pt"  # Smaller model for quicker testing
    
    logger.info(f"Testing loading Gemma-3 model: {model_name}")
    
    # First, load and inspect the config
    logger.info("Loading model configuration...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    logger.info(f"Config type: {type(config).__name__}")
    logger.info(f"Config attributes: {dir(config)}")
    
    # Check for nested text_config structure (Gemma-3 specific)
    if hasattr(config, 'text_config'):
        logger.info("Found nested text_config in Gemma config")
        logger.info(f"text_config attributes: {dir(config.text_config)}")
        
        # Copy attributes from text_config to top level
        for attr_name in dir(config.text_config):
            # Skip private attributes and methods
            if not attr_name.startswith('_') and not callable(getattr(config.text_config, attr_name)):
                if not hasattr(config, attr_name):
                    value = getattr(config.text_config, attr_name)
                    logger.info(f"Copying {attr_name}={value} from text_config to main config")
                    setattr(config, attr_name, value)
    
    # Get the vocab size from tokenizer if not present
    if not hasattr(config, 'vocab_size'):
        # Get the vocab size from the tokenizer instead
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        vocab_size = len(tokenizer.get_vocab())
        logger.info(f"Setting vocab_size to {vocab_size} from tokenizer")
        # Add the vocab_size attribute to the config
        config.vocab_size = vocab_size
    
    # Needed attributes for Gemma3 model - only used if not found in config
    required_attributes = {
        'vocab_size': 262145,  # Default value from tokenizer
        'hidden_size': 5376,   # From HF config text_config.hidden_size
        'intermediate_size': 21504,  # From HF config text_config.intermediate_size
        'num_hidden_layers': 62,  # From HF config text_config.num_hidden_layers
        'num_attention_heads': 32,   # From HF config text_config.num_attention_heads
        'num_key_value_heads': 16,  # From HF config text_config.num_key_value_heads
        'rms_norm_eps': 1e-6,
        'rope_theta': 10000.0,
        'attention_bias': False,
        'tie_word_embeddings': False
    }
    
    # Add any missing required attributes
    for attr, value in required_attributes.items():
        if not hasattr(config, attr):
            logger.warning(f"Config is missing {attr} attribute! Setting to {value}")
            setattr(config, attr, value)
        else:
            logger.info(f"Config already has {attr} = {getattr(config, attr)}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Load model with minimal kwargs
    logger.info("Loading model...")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "config": config,  # Use our modified config
    }
    
    logger.info(f"Model kwargs: {model_kwargs}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model type: {type(model).__name__}")
        
        # Test inference with a simple prompt
        logger.info("Testing inference...")
        input_text = "What is machine learning?"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Prompt: {input_text}")
        logger.info(f"Response: {response}")
        
        logger.info("Test completed successfully!")
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Trying a different approach...")
        
        # Try a more direct approach with fewer arguments
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        logger.info("Model loaded successfully with alternate method!")
        logger.info(f"Model type: {type(model).__name__}")

if __name__ == "__main__":
    main() 