"""
Centralized FxT model loading.

Usage:
    from src.eval.model_loader import load_fxt_model

    model, tokenizer = load_fxt_model("/path/to/checkpoint", device="cuda")
"""
import json
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoTokenizer
from src.model.modern_fxt import FxTTransformerLM


def load_fxt_model(model_path: str, device: str = "cuda"):
    """
    Load an FxT model + tokenizer from a checkpoint directory.

    The directory must contain:
      - config.json
      - model.safetensors
      - tokenizer files (tokenizer.json / tokenizer_config.json)

    Returns:
        (model, tokenizer)
    """
    model_dir = Path(model_path)
    config = json.load(open(model_dir / "config.json"))

    print(f"[model_loader] path={model_path}")

    # Build model from config
    class _Args:
        pass

    model_args = _Args()
    for k, v in config.items():
        setattr(model_args, k, v)

    model = FxTTransformerLM(model_args)
    state_dict = load_file(str(model_dir / "model.safetensors"))
    # state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"[model_loader] Model loaded on {device}. Params: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer, config
