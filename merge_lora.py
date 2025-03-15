#!/usr/bin/env python
# coding=utf-8

"""
Utility script to merge LoRA adapters with the base model for deployment.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="google/gemma-3-27b-pt",
        help="Path to the base model or Hugging Face model ID",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the merged model",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Save in half precision (fp16)",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="Save using safetensors format",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading base model from {args.base_model_path}...")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16 if args.half_precision else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
    )
    
    # Load LoRA configuration
    print(f"Loading LoRA adapter from {args.adapter_path}...")
    peft_config = PeftConfig.from_pretrained(args.adapter_path)
    
    # Load LoRA adapter
    print("Applying adapter to base model...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    # Merge LoRA weights with base model
    print("Merging weights...")
    model = model.merge_and_unload()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save merged model
    print(f"Saving merged model to {args.output_dir}...")
    model.save_pretrained(
        args.output_dir,
        safe_serialization=args.use_safetensors,
    )
    tokenizer.save_pretrained(args.output_dir)
    
    print("Merge completed successfully!")

if __name__ == "__main__":
    main() 