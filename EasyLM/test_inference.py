import logging
import jax
import jax.numpy as jnp
import mlxu
from transformers import AutoTokenizer

from EasyLM.models.llama.llama_config import create_llama_flags
create_llama_flags()

def main(argv):

    print("Test Inference Script starts.")
    logging.basicConfig(level=logging.INFO)

    from EasyLM.models.llama.llama_serve import ModelServer

    # Initialize model server
    server = ModelServer()

    # Format prompt using chat template
    test_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Tell me a short story about a cat.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    response = server.generate([test_prompt], temperature=1.0)
    print("\nInput: Tell me a short story about a cat.")
    print("\nOutput:", response[0])

    print ("=====================")

    test_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Tell me a scary story.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    response = server.generate([test_prompt], temperature=1.0)
    print("\nInput: Tell me a scary story.")
    print("\nOutput:", response[0])

if __name__ == '__main__':
    # print all command line arguments
    import sys
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    mlxu.run(main)
