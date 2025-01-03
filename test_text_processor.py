import yaml
from transformers import AutoTokenizer
from EasyLM.data import TextProcessor
import mlxu

def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    # Create config with template
    config = mlxu.config_dict()
    config.template = """
sequence:
  - no_loss: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>"
  - no_loss: "{instruction} {input}<|eot_id|>"
  - with_loss: "<|start_header_id|>assistant<|end_header_id|>{output}<|eot_id|>"
"""
    
    # Create processor
    processor = TextProcessor(config, tokenizer)
    
    # Test example
    example = {
        "instruction": "Classify the sentiment of this text:",
        "input": "I love this movie!",
        "output": "The sentiment is positive."
    }
    
    # Process example
    tokens, loss_masks = processor(example)
    
    # Print results
    print("\nTokens:", tokens)
    print("\nLoss masks:", loss_masks)
    print("\nDecoded tokens:", tokenizer.decode(tokens))
    print("\nToken/mask pairs:")
    for t, m in zip(tokens, loss_masks):
        token_text = tokenizer.decode([t])
        print(f"Token: {token_text:20} ID: {t:6d} Loss: {m}")

if __name__ == "__main__":
    main()
