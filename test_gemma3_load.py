#!/usr/bin/env python
# coding=utf-8

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    model_name = "google/gemma-3-2b-pt"  # Smaller model for quicker testing
    
    logger.info(f"Testing loading Gemma-3 model: {model_name}")
    
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
        # Note: We're not adding use_cache param since Gemma-3 doesn't support it
    }
    
    logger.info(f"Model kwargs: {model_kwargs}")
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

if __name__ == "__main__":
    main() 