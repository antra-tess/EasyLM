#!/usr/bin/env python
# coding=utf-8

"""
Inference script for testing the fine-tuned Gemma model.
Supports using either the LoRA adapter with the base model or a merged model.
Optimized for A100 GPUs.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="google/gemma-3-27b-pt",
        help="Path to base model or HF model ID",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (if using adapter)",
    )
    parser.add_argument(
        "--merged_model_path",
        type=str,
        default=None,
        help="Path to merged model (if not using adapter)",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit precision",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit precision",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for non-interactive mode",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use when not quantizing",
    )
    return parser.parse_args()

def format_prompt(instruction, user_name="User"):
    """Format the instruction into the template used for training"""
    return f"{instruction}\n<msg username=\"{user_name}\">"

def clear_gpu_memory():
    """Free up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def main():
    args = parse_args()
    
    # Check that we have either adapter_path or merged_model_path
    if args.adapter_path is None and args.merged_model_path is None:
        raise ValueError("Either --adapter_path or --merged_model_path must be provided")
    
    # Clear GPU memory
    clear_gpu_memory()
    
    # Load tokenizer
    if args.merged_model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.merged_model_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    
    # Ensure the tokenizer has the pad token set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization settings or GPU memory optimization
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        # Use GPU memory efficient loading if not quantizing
        model_kwargs["device_map"] = "auto"
        model_kwargs["max_memory"] = {0: f"{int(args.gpu_memory_utilization * 100)}%"}
    
    # Load model
    if args.merged_model_path:
        print(f"Loading merged model from {args.merged_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.merged_model_path,
            **model_kwargs
        )
    else:
        print(f"Loading base model from {args.base_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            **model_kwargs
        )
        
        print(f"Loading adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        
        # Optional: merge adapter with model for faster inference
        if not args.load_in_4bit and not args.load_in_8bit:
            print("Merging adapter weights for faster inference...")
            model = model.merge_and_unload()
    
    # Create a text streamer for streaming output
    streamer = TextStreamer(tokenizer, skip_special_tokens=True)
    
    # Set up generation parameters
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }
    
    if args.interactive:
        # Interactive mode
        print("\n===== Interactive Mode =====")
        print("Enter your prompt (or 'q' to quit)")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ("q", "quit", "exit"):
                break
            
            # Format the prompt
            formatted_prompt = format_prompt(user_input)
            
            # Tokenize the prompt
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            # Generate
            print("\nAssistant: ", end="")
            with torch.inference_mode():
                _ = model.generate(**inputs, **generation_config)
            
            # Print a newline after generation
            print()
    else:
        # Single prompt mode
        if not args.prompt:
            args.prompt = "Write a short poem about a cat who dreams of being a tiger."
        
        # Format the prompt
        formatted_prompt = format_prompt(args.prompt)
        
        # Print the prompt
        print(f"\nPrompt: {args.prompt}")
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate
        print("\nAssistant: ", end="")
        with torch.inference_mode():
            _ = model.generate(**inputs, **generation_config)
        
        # Print a newline after generation
        print()
    
    # Clear GPU memory when done
    del model
    clear_gpu_memory()

if __name__ == "__main__":
    main() 