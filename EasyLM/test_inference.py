import logging
import jax
import jax.numpy as jnp
import mlxu
from transformers import AutoTokenizer

from EasyLM.models.llama.llama_serve import ModelServer
from EasyLM.models.llama.llama_config import create_llama_flags
from EasyLM.models.llama.llama_config import FLAGS

def main(argv):


    logging.basicConfig(level=logging.INFO)
    
    # Initialize model server
    server = ModelServer()
    
    # Run inference on fixed text
    test_text = ["Tell me a short story about a cat."]
    response = server.generate(test_text, temperature=1.0)
    
    print("\nInput:", test_text[0])
    print("\nOutput:", response[0])

if __name__ == '__main__':
    FLAGS2 = create_llama_flags()
    print(FLAGS)
    print(FLAGS2)

    mlxu.run(main)


