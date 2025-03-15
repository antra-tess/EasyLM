#!/usr/bin/env python
# coding=utf-8

import torch
import logging
import os
import time
import json

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Helper function to patch the config
def patch_gemma3_config(config):
    """Patch Gemma3Config to add required attributes from text_config."""
    # First make sure vocab_size is set
    if not hasattr(config, "vocab_size"):
        setattr(config, "vocab_size", 262144)
        logger.info("Manually set vocab_size to 262144")
    
    # Essential attributes from text_config
    if hasattr(config, "text_config"):
        text_config = config.text_config
        if isinstance(text_config, dict):
            essential_attrs = [
                "hidden_size", "intermediate_size", "num_attention_heads", 
                "num_hidden_layers", "num_key_value_heads", "head_dim", 
                "query_pre_attn_scalar"
            ]
            
            for attr in essential_attrs:
                if attr in text_config and not hasattr(config, attr):
                    setattr(config, attr, text_config[attr])
                    logger.info(f"Copied {attr}={text_config[attr]} from text_config")
    
    # Verify essential attributes
    for attr in ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads"]:
        if not hasattr(config, attr):
            logger.warning(f"Config still missing essential attribute: {attr}")
    
    return config

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
    
    # APPROACH 1: Use transformers.pipeline with custom config patching
    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # Load and patch config first
        logger.info("Loading and patching config...")
        config = AutoConfig.from_pretrained(
            "google/gemma-3-27b-pt", 
            token=hf_token,
            trust_remote_code=True
        )
        
        # Print the config structure for debugging
        config_dict = config.to_dict()
        logger.info(f"Config top-level keys: {list(config_dict.keys())}")
        if 'text_config' in config_dict:
            logger.info(f"Text config keys: {list(config_dict['text_config'].keys())}")
        
        # Patch the config
        config = patch_gemma3_config(config)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-3-27b-pt",
            trust_remote_code=True,
            token=hf_token
        )
        
        # Try to directly monkey-patch the model class before initialization
        try:
            # Direct monkey patching of the Gemma3 class
            logger.info("Attempting to monkey-patch Gemma3TextModel.__init__...")
            
            original_init = transformers.models.gemma3.modeling_gemma3.Gemma3TextModel.__init__
            
            def patched_init(self, config, *args, **kwargs):
                # First ensure config has vocab_size
                if not hasattr(config, "vocab_size"):
                    config.vocab_size = 262144
                    logger.info("Patched vocab_size in __init__")
                
                # If text_config exists, copy essential attributes
                if hasattr(config, "text_config"):
                    if isinstance(config.text_config, dict):
                        for key, value in config.text_config.items():
                            if not hasattr(config, key):
                                setattr(config, key, value)
                
                # Call original init
                return original_init(self, config, *args, **kwargs)
            
            # Replace the init method
            transformers.models.gemma3.modeling_gemma3.Gemma3TextModel.__init__ = patched_init
            logger.info("Successfully monkey-patched Gemma3TextModel.__init__")
        
        except Exception as e:
            logger.warning(f"Could not monkey-patch Gemma3TextModel: {e}")
        
        # Load model with patched config
        logger.info("Loading model with patched config...")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-27b-pt",
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
            low_cpu_mem_usage=True
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
            
        # Try alternative embedding path
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            logger.info(f"Embed tokens shape: {model.model.embed_tokens.weight.shape}")
        
        # Test inference
        logger.info("Running inference...")
        input_text = "Hello, my name is"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Move to appropriate device
        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        
        # Generate
        outputs = model.generate(**inputs, max_new_tokens=20)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
        
        return True
        
    except Exception as e:
        logger.error(f"Direct approach failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # APPROACH 2: Create a custom model class that handles the missing attributes
    try:
        logger.info("Trying custom model class approach...")
        import transformers
        from transformers import AutoConfig, AutoTokenizer
        
        # Create a custom wrapper for the Gemma3ForCausalLM class
        from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM, Gemma3TextModel
        
        # First patch the Gemma3TextModel class
        class PatchedGemma3TextModel(Gemma3TextModel):
            def __init__(self, config):
                # Ensure vocab_size is set before parent initialization
                if not hasattr(config, "vocab_size"):
                    config.vocab_size = 262144
                
                # Copy text_config attributes if needed
                if hasattr(config, "text_config") and isinstance(config.text_config, dict):
                    for key, value in config.text_config.items():
                        if not hasattr(config, key):
                            setattr(config, key, value)
                
                super().__init__(config)
        
        # Then create a patched causal LM model
        class PatchedGemma3ForCausalLM(Gemma3ForCausalLM):
            def __init__(self, config):
                # Ensure vocab_size is set before parent initialization
                if not hasattr(config, "vocab_size"):
                    config.vocab_size = 262144
                
                # Save original class
                original_model_class = Gemma3TextModel
                
                # Replace with our patched version temporarily
                transformers.models.gemma3.modeling_gemma3.Gemma3TextModel = PatchedGemma3TextModel
                
                # Call parent init
                super().__init__(config)
                
                # Restore original class
                transformers.models.gemma3.modeling_gemma3.Gemma3TextModel = original_model_class
        
        # Register model with Auto classes
        transformers.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING.register(transformers.models.gemma3.Gemma3Config, PatchedGemma3ForCausalLM)
        
        # Load the config
        config = AutoConfig.from_pretrained(
            "google/gemma-3-27b-pt", 
            token=hf_token,
            trust_remote_code=True
        )
        
        # Patch config directly
        patch_gemma3_config(config)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-3-27b-pt",
            trust_remote_code=True,
            token=hf_token
        )
        
        # Load model with our patched model class
        logger.info("Loading model with patched model class...")
        model = PatchedGemma3ForCausalLM.from_pretrained(
            "google/gemma-3-27b-pt",
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
            low_cpu_mem_usage=True
        )
        
        # Check embedding layer
        logger.info("Checking embedding layer")
        if hasattr(model, 'get_input_embeddings'):
            embed = model.get_input_embeddings()
            if embed is not None and hasattr(embed, 'weight'):
                logger.info(f"Input embeddings weight shape: {embed.weight.shape}")
            else:
                logger.info("Input embeddings has no weight attribute")
        
        # Test inference
        logger.info("Running inference...")
        input_text = "Hello, my name is"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Move to appropriate device
        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        
        # Generate
        outputs = model.generate(**inputs, max_new_tokens=20)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
        
        return True
        
    except Exception as e:
        logger.error(f"Custom model class approach failed: {e}")
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