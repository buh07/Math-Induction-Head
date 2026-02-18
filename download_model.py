#!/usr/bin/env python3
"""
Download a HuggingFace model for the induction heads experiment.

Usage:
    python download_model.py --model meta-llama/Llama-2-7b-hf
    python download_model.py --model meta-llama/Llama-3-8b-instruct
    python download_model.py --model gpt2
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import login, hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def authenticate_huggingface():
    """Authenticate with HuggingFace using stored token or prompt."""
    try:
        # Try to login using stored token if available
        login()
        print("✓ Authenticated with HuggingFace")
    except Exception as e:
        print(f"Could not auto-authenticate: {e}")
        print("\nTo authenticate, you have two options:")
        print("1. Use HF token: huggingface-cli login")
        print("2. Set HF_TOKEN environment variable: export HF_TOKEN=hf_...")
        print("\nRetrying without authentication (will fail for gated models)...")


def download_model(model_name: str, token: str = None, use_auth_token: bool = True):
    """
    Download model and tokenizer from HuggingFace.
    
    Args:
        model_name: Model identifier (e.g., 'meta-llama/Llama-2-7b-hf')
        token: Optional HF token
        use_auth_token: Whether to use authentication
    """
    print(f"Downloading model: {model_name}")
    print("=" * 60)
    
    try:
        # Download tokenizer
        print(f"\n1. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token or use_auth_token,
            trust_remote_code=True,
        )
        print(f"   ✓ Tokenizer saved to cache")
        
        # Download model
        print(f"\n2. Downloading model (this may take several minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token or use_auth_token,
            trust_remote_code=True,
            torch_dtype="auto",  # Auto-detect dtype
        )
        print(f"   ✓ Model saved to cache")
        
        # Get cache location
        from transformers.utils import TRANSFORMERS_CACHE
        model_cache = Path(TRANSFORMERS_CACHE) / "models--" + model_name.replace("/", "--")
        
        print(f"\n3. Model details:")
        print(f"   Model ID: {model_name}")
        print(f"   Model type: {model.config.model_type}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Cache location: {model_cache}")
        
        print(f"\n✓ Download complete!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. If gated model, ensure you have accepted license at huggingface.co/{model_name}")
        print(f"2. If auth error, run: huggingface-cli login")
        print(f"3. Or set token: export HF_TOKEN=hf_...")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace model")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model identifier to download"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token (if not set globally)"
    )
    
    args = parser.parse_args()
    
    # Authenticate
    authenticate_huggingface()
    
    # Download model
    success = download_model(args.model, token=args.token)
    
    if success:
        print(f"\nYou can now use the model with:")
        print(f"  from transformers import AutoModelForCausalLM")
        print(f"  model = AutoModelForCausalLM.from_pretrained('{args.model}')")
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
