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
    
    logger.info(f"Loading model from {model_name}")
    
    # Configure model loading
    compute_dtype = torch.bfloat16  # Use bfloat16 for more efficient inference
    
    # Set up model loading parameters
    model_kwargs = {
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
        "token": hf_token,
        "device_map": "auto",  # Let transformers handle multi-GPU placement
        "max_memory": {i: "40GiB" for i in range(torch.cuda.device_count())},  # Allocate memory for each GPU
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

if __name__ == "__main__":
    test_gemma_inference() 