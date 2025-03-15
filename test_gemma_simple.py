#!/usr/bin/env python
# coding=utf-8

import torch
import logging
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def test_gemma_pipeline():
    """
    Simple test using pipeline API to check if the base Gemma-3 model works for inference without LoRA.
    Optimized for running on 2xA100 GPUs.
    """
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN environment variable not set. You may encounter authentication issues.")
    
    model_name = "google/gemma-3-27b-pt"  # 27B parameter model
    
    logger.info(f"Using pipeline API to load {model_name}")
    logger.info(f"Found {torch.cuda.device_count()} CUDA devices")
    
    # First, let's try loading just the tokenizer separately to check vocab_size
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        token=hf_token,
    )
    logger.info(f"Tokenizer loaded with vocab_size: {tokenizer.vocab_size}")
    
    # Create text generation pipeline
    logger.info("Creating text generation pipeline...")
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
        model_kwargs={"low_cpu_mem_usage": True},
    )
    
    # Get model from pipeline to inspect it
    logger.info("Pipeline created, extracting model to inspect embeddings...")
    model = pipe.model
    
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
    
    # Run inference
    logger.info("Running inference...")
    test_input = "Hello, my name is"
    
    try:
        result = pipe(
            test_input,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
        )
        
        logger.info(f"Generated text: {result[0]['generated_text']}")
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
    test_gemma_pipeline() 