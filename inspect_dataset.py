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
    config.dataset.type = 'json'
    config.dataset.text_processor.template = open('templates/test_chat.yaml').read()
    config.dataset.json_dataset.path = 'test_conversations.json'
    config.dataset.json_dataset.seq_length = 1024
    config.dataset.json_dataset.batch_size = 2  # Small batch for inspection

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

    # Load dataset
    dataset = DatasetFactory.load_dataset(config.dataset, tokenizer)

    print("Inspecting dataset batches...")
    print("=" * 80)

    # Look at just a few examples in detail
    print("\n=== Detailed Example Inspection ===\n")
    for batch_idx, (batch, metrics) in enumerate(dataset):
        if batch_idx >= 1:  # Just look at first batch
            break
            
        # For first few sequences in batch
        for seq_idx in range(min(2, config.dataset.huggingface_dataset.batch_size)):
            print(f"\nExample {seq_idx + 1}:")
            print("=" * 80)
            
            input_tokens = batch['input_tokens'][seq_idx]
            target_tokens = batch['target_tokens'][seq_idx]
            loss_masks = batch['loss_masks'][seq_idx]
            
            # Show statistics
            print(f"\nSequence Statistics:")
            print(f"  Total length: {len(input_tokens)}")
            print(f"  Tokens with loss: {np.sum(loss_masks)}")
            print(f"  Loss percentage: {np.mean(loss_masks)*100:.1f}%")
            
            # Show the full sequence broken into segments by loss mask
            print(f"\nSequence Breakdown:")
            current_mask = loss_masks[0]
            current_text = ""
            for i, (inp, mask) in enumerate(zip(input_tokens, loss_masks)):
                if mask != current_mask:
                    print(f"  {'[LOSS]' if current_mask else '[ -- ]'} {current_text}")
                    current_text = ""
                    current_mask = mask
                current_text += tokenizer.decode([inp])
            print(f"  {'[LOSS]' if current_mask else '[ -- ]'} {current_text}")
            
            # Show token-level details for a small window
            print(f"\nDetailed Token View (first 10 transitions):")
            transitions_shown = 0
            for i, (inp, tgt, mask) in enumerate(zip(input_tokens, target_tokens, loss_masks)):
                if i > 0 and loss_masks[i] != loss_masks[i-1]:
                    print(f"\n  Position {i}:")
                    print(f"    Input:  {inp} -> {tokenizer.decode([inp])}")
                    print(f"    Target: {tgt} -> {tokenizer.decode([tgt])}")
                    print(f"    Mask:   {mask}")
                    transitions_shown += 1
                    if transitions_shown >= 10:
                        break
            
            print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
