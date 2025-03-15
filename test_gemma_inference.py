#!/usr/bin/env python
# coding=utf-8

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """
    model_name = "google/gemma-3-27b-pt"  # Use the same model path as in your training script
    
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model from {model_name}")
    
    # Configure model loading
    compute_dtype = torch.bfloat16  # Use bfloat16 for more efficient inference
    
    # Set up model loading parameters
    model_kwargs = {
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
        "attn_implementation": "eager",  # Recommended for Gemma models
        "device_map": "auto",  # Use all available GPUs
    }
    
    logger.info(f"Loading with kwargs: {model_kwargs}")
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Check and log embedding layer status
    if hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'weight'):
        embed_shape = model.model.embed_tokens.weight.shape
        logger.info(f"Embed tokens weight shape: {embed_shape}")
    
    # Test with a simple input
    test_input = "Hello, my name is"
    
    logger.info(f"Running inference with input: '{test_input}'")
    
    # Tokenize input
    input_ids = tokenizer(test_input, return_tensors="pt").to(model.device)
    
    # Generate output
    with torch.no_grad():
        try:
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

if __name__ == "__main__":
    test_gemma_inference() 