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
    
    # Check for vocabulary size in config
    if not hasattr(config, 'vocab_size'):
        logger.warning("Config is missing vocab_size attribute!")
        # Get the vocab size from the tokenizer instead
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        vocab_size = len(tokenizer.get_vocab())
        logger.info(f"Setting vocab_size to {vocab_size} from tokenizer")
        # Add the vocab_size attribute to the config
        config.vocab_size = vocab_size
    
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