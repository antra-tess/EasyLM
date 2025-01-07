import os
import logging
import numpy as np
from ml_collections import ConfigDict
import mlxu
from transformers import AutoTokenizer
from EasyLM.data import DatasetFactory

def print_tokens(tokenizer, tokens, loss_mask=None, prefix=""):
    """Pretty print tokens with their text and optionally loss masks."""
    text = tokenizer.decode(tokens)
    print(f"{prefix}Text: {text}")
    print(f"{prefix}Tokens: {tokens}")
    if loss_mask is not None:
        print(f"{prefix}Mask: {loss_mask}")
        # Show which text parts have loss applied
        parts = []
        current_text = ""
        current_mask = loss_mask[0]
        for token, mask in zip(tokens, loss_mask):
            if mask != current_mask:
                parts.append((current_text, current_mask))
                current_text = ""
                current_mask = mask
            current_text += tokenizer.decode([token])
        parts.append((current_text, current_mask))
        print(f"{prefix}Text with loss regions:")
        for text, mask in parts:
            print(f"{prefix}  {'[LOSS]' if mask else '[ -- ]'} {text}")
    print()

def main():
    # Load the same config used in training
    config = mlxu.config_dict()
    config.dataset = DatasetFactory.get_default_config()
    config.dataset.type = 'huggingface'
    config.dataset.text_processor.template = """
sequence:
  - no_loss: "{instruction}\\n{input}\\n"
  - with_loss: "{output}\\n"
"""
    config.dataset.huggingface_dataset.path = "tatsu-lab/alpaca"
    config.dataset.huggingface_dataset.name = ""
    config.dataset.huggingface_dataset.split = "train"
    config.dataset.huggingface_dataset.seq_length = 1024
    config.dataset.huggingface_dataset.batch_size = 2  # Small batch for inspection

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Load dataset
    dataset = DatasetFactory.load_dataset(config.dataset, tokenizer)

    print("Inspecting dataset batches...")
    print("=" * 80)

    # Look at first few batches
    for batch_idx, (batch, metrics) in enumerate(dataset):
        if batch_idx >= 3:  # Just look at first 3 batches
            break

        print(f"Batch {batch_idx}:")
        print("-" * 40)
        
        # For each sequence in batch
        for seq_idx in range(config.dataset.huggingface_dataset.batch_size):
            print(f"Sequence {seq_idx}:")
            
            # Show input tokens
            input_tokens = batch['input_tokens'][seq_idx]
            print_tokens(tokenizer, input_tokens, prefix="  Input  | ")
            
            # Show target tokens
            target_tokens = batch['target_tokens'][seq_idx]
            print_tokens(tokenizer, target_tokens, prefix="  Target | ")
            
            # Show loss masks and their effect
            loss_masks = batch['loss_masks'][seq_idx]
            print(f"  Loss Mask | {loss_masks}")
            
            # Verify alignment
            print("\n  Token Alignment:")
            for i, (inp, tgt, mask) in enumerate(zip(input_tokens, target_tokens, loss_masks)):
                if inp != tgt:
                    print(f"    Position {i}:")
                    print(f"      Input:  {inp} -> {tokenizer.decode([inp])}")
                    print(f"      Target: {tgt} -> {tokenizer.decode([tgt])}")
                    print(f"      Mask:   {mask}")
            
            print("\n" + "-" * 40)
        print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
