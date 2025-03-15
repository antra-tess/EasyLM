#!/usr/bin/env python
# coding=utf-8

import torch
import logging
import os
import json
import time
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def test_gemma_inference(prompt_file=None, num_completions=5):
    """
    Comprehensive approach to load Gemma-3 model addressing all nested attributes.
    Can generate multiple completions for a prompt loaded from a file.
    
    Args:
        prompt_file: Path to a text file containing the prompt
        num_completions: Number of different completions to generate
    """
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN environment variable not set. You may encounter authentication issues.")
    
    logger.info(f"Found {torch.cuda.device_count()} CUDA devices")
    
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # 1. Create a custom Gemma3Config class that addresses the nested structure
        from transformers import Gemma3Config as OriginalGemma3Config
        from transformers import AutoTokenizer
        
        class CustomGemma3Config(OriginalGemma3Config):
            """Custom config that automatically copies all text_config attributes to root level."""
            
            @classmethod
            def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
                config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
                
                # Get tokenizer to extract vocab_size if needed
                try:
                    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
                    if hasattr(tokenizer, 'vocab_size'):
                        logger.info(f"Got vocab_size from tokenizer: {tokenizer.vocab_size}")
                        config.vocab_size = tokenizer.vocab_size
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer: {e}")
                
                # Extract and copy all text_config attributes
                config_dict = config.to_dict()
                if 'text_config' in config_dict and isinstance(config_dict['text_config'], dict):
                    for key, value in config_dict['text_config'].items():
                        logger.info(f"Copying attribute from text_config: {key}={value}")
                        setattr(config, key, value)
                
                # Manually set sliding_window_pattern if needed
                if hasattr(config, "sliding_window") and not hasattr(config, "sliding_window_pattern"):
                    # Default pattern that seems common in Gemma models
                    setattr(config, "sliding_window_pattern", 6)
                    logger.info(f"Manually added sliding_window_pattern=6 based on common patterns")
                
                return config
        
        model_name = "google/gemma-3-27b-pt"
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
        )
        logger.info(f"Tokenizer loaded with vocab_size: {tokenizer.vocab_size}")
        
        # Load custom config that auto-copies all text_config attributes
        logger.info("Loading and patching config with custom class")
        config = CustomGemma3Config.from_pretrained(model_name, token=hf_token)
        
        # Log all available config attributes for debugging
        config_attrs = [attr for attr in dir(config) if not attr.startswith('_') and not callable(getattr(config, attr))]
        logger.info(f"Config now has attributes: {config_attrs}")
        
        # Double-check critical attributes
        critical_attrs = [
            "vocab_size", "hidden_size", "intermediate_size", "num_attention_heads", 
            "num_hidden_layers", "sliding_window", "sliding_window_pattern"
        ]
        for attr in critical_attrs:
            if hasattr(config, attr):
                logger.info(f"✅ Config has {attr}: {getattr(config, attr)}")
            else:
                logger.warning(f"❌ Config missing critical attribute: {attr}")
        
        # Get the correct model class
        from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
        
        # Load model with fully patched config
        logger.info("Loading model with fully patched config...")
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token=hf_token,
            # Try loading without device_map first to avoid meta device issues
            # We'll move it to the appropriate device after loading
        )
        
        # After successful loading, check embedding layer
        logger.info("Checking embedding layer")
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            if hasattr(model.model.embed_tokens, 'weight'):
                embed_shape = model.model.embed_tokens.weight.shape
                embed_device = model.model.embed_tokens.weight.device
                logger.info(f"Embed tokens weight shape: {embed_shape}")
                logger.info(f"Embed tokens device: {embed_device}")
            else:
                logger.info("Embed tokens has no weight attribute")
        
        # Load prompt from file if provided, otherwise use default
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                test_input = f.read().strip()
            logger.info(f"Loaded prompt from file: {prompt_file}")
            logger.info(f"Prompt preview: '{test_input[:100]}...' (truncated)")
        else:
            test_input = "Hello, my name is"
            logger.info(f"Using default prompt: '{test_input}'")
        
        # Tokenize input
        inputs = tokenizer(test_input, return_tensors="pt")
        
        # Move model to appropriate device if not already there
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Moving model to {device}")
            model = model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate multiple completions
        logger.info(f"Generating {num_completions} different completions...")
        all_outputs = []
        
        for i in range(num_completions):
            logger.info(f"Starting generation #{i+1}...")
            # Use different random seeds for variety
            torch.manual_seed(42 + i)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,  # Reasonable length for a short story
                do_sample=True,
                temperature=0.9,     # Slightly higher temperature for creativity
                top_p=0.99,          # More focused sampling
                top_k=250,            # Reasonable diversity
            )
            
            # Decode and save output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            all_outputs.append(generated_text)
            logger.info(f"Completion #{i+1}: '{generated_text}'")
        
        logger.info("✅ All generations completed successfully!")
        
        # Save all completions to a file
        output_dir = Path("./generated_completions")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = output_dir / f"gemma_completions_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, output in enumerate(all_outputs):
                f.write(f"COMPLETION #{i+1}:\n{output}\n\n")
                f.write("-" * 40 + "\n\n")
        
        logger.info(f"All completions saved to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Final approach failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run Gemma-3 inference with multiple completions')
    parser.add_argument('--prompt_file', type=str, help='Path to a text file containing the prompt')
    parser.add_argument('--num_completions', type=int, default=5, help='Number of completions to generate')
    
    args = parser.parse_args()
    
    start_time = time.time()
    success = test_gemma_inference(prompt_file=args.prompt_file, num_completions=args.num_completions)
    end_time = time.time()
    
    if success:
        logger.info(f"✅ Test completed successfully in {end_time - start_time:.2f} seconds")
    else:
        logger.error(f"❌ Test failed after {end_time - start_time:.2f} seconds") 