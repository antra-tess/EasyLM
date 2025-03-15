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
    Direct approach to load Gemma-3 model based on Hugging Face team recommendation.
    Using Gemma3ForCausalLM directly instead of AutoModelForCausalLM.
    """
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN environment variable not set. You may encounter authentication issues.")
    
    logger.info(f"Found {torch.cuda.device_count()} CUDA devices")
    
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # Import specific model class instead of using Auto classes
        from transformers import Gemma3ForCausalLM, AutoTokenizer, Gemma3Config
        
        model_name = "google/gemma-3-27b-pt"
        
        # Load tokenizer first
        logger.info(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
            token=hf_token,
        )
        logger.info(f"Tokenizer loaded with vocab_size: {tokenizer.vocab_size}")
        
        # Load config and manually patch it
        logger.info("Loading and patching config")
        config = Gemma3Config.from_pretrained(model_name, token=hf_token)
        
        # Manually add vocab_size to config
        if not hasattr(config, "vocab_size"):
            vocab_size = tokenizer.vocab_size
            logger.info(f"Setting vocab_size to {vocab_size}")
            config.vocab_size = vocab_size
        
        # Add other essential attributes from text_config if needed
        config_dict = config.to_dict()
        if 'text_config' in config_dict:
            text_config = config_dict['text_config']
            essential_attrs = [
                "hidden_size", "intermediate_size", "num_attention_heads", 
                "num_hidden_layers", "num_key_value_heads"
            ]
            for key, value in text_config.items():
                if not hasattr(config, key) and key in essential_attrs:
                    logger.info(f"Copying essential attribute {key}={value} from text_config")
                    setattr(config, key, value)
        
        # Check config has essential attributes
        for attr in ["vocab_size", "hidden_size"]:
            if hasattr(config, attr):
                logger.info(f"Config has {attr}: {getattr(config, attr)}")
            else:
                logger.warning(f"Config is missing {attr}")
        
        # Load the model using specific model class
        logger.info("Loading model using Gemma3ForCausalLM directly...")
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
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
        
        # Try alternative path
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            if hasattr(model.model.embed_tokens, 'weight'):
                logger.info(f"Embed tokens shape: {model.model.embed_tokens.weight.shape}")
            else:
                logger.info("Embed tokens has no weight attribute")
        
        # Test inference
        test_input = "Hello, my name is"
        logger.info(f"Running inference with input: '{test_input}'")
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="pt")
        
        # Move to appropriate device
        input_device = next(model.parameters()).device
        logger.info(f"Moving inputs to device: {input_device}")
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        
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
        logger.error(f"Direct model approach failed: {e}")
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