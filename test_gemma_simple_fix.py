#!/usr/bin/env python
# coding=utf-8

import torch
import logging
import os
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
    Simplified approach to load Gemma-3 model based on community solutions.
    Avoiding meta device errors.
    """
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN environment variable not set. You may encounter authentication issues.")
    
    logger.info(f"Found {torch.cuda.device_count()} CUDA devices")
    
    # APPROACH 1: Use transformers.pipeline with specific accelerate version
    # Recent versions of accelerate (>0.21.0) might cause the meta device error
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
        
        from transformers import pipeline
        
        logger.info("Loading model with pipeline API (simplest approach)...")
        pipe = pipeline(
            "text-generation",
            model="google/gemma-3-27b-pt",  # Using smaller model first
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )
        
        # Extract model to check embeddings
        model = pipe.model
        
        # Test inference
        logger.info("Running inference with pipeline...")
        result = pipe("Hello, my name is", max_new_tokens=20)
        logger.info(f"Generated text: {result[0]['generated_text']}")
        
        # Check embedding layer
        logger.info("Checking embedding layer")
        if hasattr(model, 'get_input_embeddings'):
            embed = model.get_input_embeddings()
            if embed is not None and hasattr(embed, 'weight'):
                logger.info(f"Input embeddings weight shape: {embed.weight.shape}")
                logger.info(f"Input embeddings device: {embed.weight.device}")
            else:
                logger.info("Input embeddings has no weight attribute")
        else:
            logger.info("Could not find input embeddings method")
            
        # Try alternative embedding path
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            logger.info(f"Embed tokens shape: {model.model.embed_tokens.weight.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline approach failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # APPROACH 2: Manually download then load model
    try:
        logger.info("Trying alternative approach with manual loading...")
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # First download model files
        logger.info("Downloading model snapshot...")
        model_path = snapshot_download(
            repo_id="google/gemma-3-27b-pt",
            token=hf_token,
            local_dir="./gemma_model_cache"
        )
        
        logger.info(f"Model downloaded to {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model - with minimal parameters
        logger.info("Loading model from local path...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Check embedding layer
        logger.info("Checking embedding layer")
        if hasattr(model, 'get_input_embeddings'):
            embed = model.get_input_embeddings()
            if embed is not None and hasattr(embed, 'weight'):
                logger.info(f"Input embeddings weight shape: {embed.weight.shape}")
                logger.info(f"Input embeddings device: {embed.weight.device}")
            else:
                logger.info("Input embeddings has no weight attribute")
        else:
            logger.info("Could not find input embeddings method")
        
        # Test inference
        input_text = "Hello, my name is"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Move to appropriate device
        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        
        # Generate
        logger.info("Running inference...")
        outputs = model.generate(**inputs, max_new_tokens=20)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
        
        return True
        
    except Exception as e:
        logger.error(f"Alternative approach failed: {e}")
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
        logger.error(f"❌ All approaches failed after {end_time - start_time:.2f} seconds") 