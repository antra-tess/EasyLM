import os
import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

def download_and_examine_config(model_id="meta-llama/Llama-3.1-8B"):
    try:
        # Try direct config download first
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            token=os.environ.get("HUGGINGFACE_TOKEN")
        )
        
        # Load and print raw config
        with open(config_path, 'r') as f:
            raw_config = json.load(f)
        print("\nRaw config from HF:")
        print(json.dumps(raw_config, indent=2))
        
        print("\nAttempting to load with AutoConfig...")
        try:
            config = AutoConfig.from_pretrained(model_id)
            print("\nSuccessfully loaded config:")
            print(config)
        except Exception as e:
            print("\nError loading with AutoConfig:")
            print(e)
            
    except Exception as e:
        print(f"Error downloading config: {e}")

if __name__ == "__main__":
    download_and_examine_config()
