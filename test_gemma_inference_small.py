#!/usr/bin/env python
# coding=utf-8

import torch
import logging
import os
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
    Using the smaller 1B model which should work on most systems.
    """
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN environment variable not set. You may encounter authentication issues.")
    
    model_name = "google/gemma-3-1b-pt"  # Use the correct model name
    
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
    
    logger.info(f"Loading model from {model_name}")
    
    # Configure model loading
    compute_dtype = torch.bfloat16  # Use bfloat16 for more efficient inference
    
    # Set up model loading parameters - simpler setup for small model
    model_kwargs = {
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
        "token": hf_token,
    }
    
    logger.info(f"Loading with kwargs: {model_kwargs}")
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Check and log embedding layer status before inference
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'weight'):
        embed_shape = model.model.embed_tokens.weight.shape
        logger.info(f"Embed tokens weight shape BEFORE inference: {embed_shape}")
    else:
        logger.info("Could not find embed_tokens via model.model.embed_tokens path")
        
        # Try alternative paths
        if hasattr(model, 'get_input_embeddings'):
            embed = model.get_input_embeddings()
            if embed is not None and hasattr(embed, 'weight'):
                logger.info(f"Input embeddings weight shape: {embed.weight.shape}")
            else:
                logger.info("Input embeddings not found or has no weight attribute")
    
    # Test with a simple input
    test_input = "Hello, my name is"
    
    logger.info(f"Running inference with input: '{test_input}'")
    
    # Tokenize input
    input_ids = tokenizer(test_input, return_tensors="pt")
    
    # Move to appropriate device
    if torch.cuda.is_available():
        model = model.to("cuda")
        input_ids = {k: v.to("cuda") for k, v in input_ids.items()}
        logger.info("Using CUDA for inference")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # For Mac M-series chips
        model = model.to("mps")
        input_ids = {k: v.to("mps") for k, v in input_ids.items()}
        logger.info("Using MPS (Metal) for inference")
    else:
        logger.info("Using CPU for inference")
    
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
            
            # Check embedding shape AFTER inference
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'weight'):
                embed_shape = model.model.embed_tokens.weight.shape
                logger.info(f"Embed tokens weight shape AFTER inference: {embed_shape}")
            
        except Exception as e:
            logger.error(f"❌ Inference failed with error: {str(e)}")
            logger.error("This suggests an issue with the base model itself.")
            
            # Try to get more information about the error
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_gemma_inference() 