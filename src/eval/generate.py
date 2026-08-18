import os
import sys
import json
import argparse

import torch
import transformers
from transformers import AutoTokenizer, set_seed
from safetensors.torch import load_file

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.model.modern_fxt import FxTTransformerLM
from src.utils.data_utils import BYTE_MODEL

os.environ["WANDB_DISABLED"] = "true"
transformers.logging.set_verbosity_error()


# Every mode below is reached through the same entry point, model.generate();
# the flags select which decoding path it dispatches to internally.
MODES = {
    # no cache, no speculation -> generate_group
    "plain": dict(use_caching=False, speculative=False),
    # reuse the group KV cache across steps -> generate_group_cached
    "cached": dict(use_caching=True, speculative=False),
    # draft with the model's own MTP heads, then verify -> generate_verify
    "self_speculative": dict(use_caching=False, speculative=True),
    # draft with a separate model, then verify -> generate_verify_with_fxt
    "drafter": dict(use_caching=False, speculative=True),
}


def pargs():
    parser = argparse.ArgumentParser(
        description="Generate text with an FxT checkpoint."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Checkpoint directory (config.json + model.safetensors + tokenizer)")
    parser.add_argument("--mode", type=str, default="plain", choices=sorted(MODES),
                        help="Which decoding path model.generate() should take")
    parser.add_argument("--drafter_path", type=str, default=None,
                        help="Checkpoint for the draft model; required for --mode drafter")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text")
    parser.add_argument("--prompt_file", type=str, default=None, help="File to read the prompt from")

    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Acceptance threshold for the speculative modes")
    parser.add_argument("--candidates", type=int, default=1,
                        help="Draft tokens proposed per step in the speculative modes")

    parser.add_argument("--show_tokenization", action="store_true",
                        help="Print the learned patch boundaries for the prompt")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.mode == "drafter" and args.drafter_path is None:
        parser.error("--mode drafter requires --drafter_path")
    if args.prompt is None and args.prompt_file is None:
        parser.error("provide --prompt or --prompt_file")
    return args


def load_model_and_tokenizer(model_path, args, device):
    """Build an FxT model from its saved config.json and load its weights."""
    model_config = json.load(open(f"{model_path}/config.json"))

    class ModelArgs:
        pass

    model_args = ModelArgs()
    for key, value in model_config.items():
        setattr(model_args, key, value)
    model_args.model_path = model_path
    model_args.cache_dir = args.cache_dir
    if getattr(model_args, "tokenizer_path", BYTE_MODEL) != BYTE_MODEL:
        model_args.script_tokens = []

    model = FxTTransformerLM(model_args)
    model.load_state_dict(load_file(f"{model_path}/model.safetensors"))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        extra_ids=0,
        cache_dir=args.cache_dir,
        add_eos_token=False,
    )
    return model, tokenizer, model_config


def tokenize(model, tokenizer, prompt, lang_token_id="<en>", device="cpu"):
    """Show how the model segments the prompt into patches, marked with '|'."""
    # ByT5 has no BOS and *appends* eos; match generate() and feed pure content
    # bytes so input_ids aligns 1:1 with the boundary predictions.
    tokenized_text = tokenizer(prompt, add_special_tokens=False)
    model_input = {
        "input_ids": torch.tensor(tokenized_text["input_ids"], device=device).unsqueeze(0),
        "attention_mask": torch.tensor(tokenized_text["attention_mask"], device=device).unsqueeze(0),
    }
    with torch.no_grad():
        _, overall_stats, _ = model(model_input, task="tokenization2", script_id=lang_token_id)
    boundaries = overall_stats["hard_boundaries"].float().cpu().squeeze().numpy()
    input_ids = model_input["input_ids"][0].cpu().numpy()

    # Each byte must line up with its own boundary decision. A mismatch here is
    # exactly what silently shifted/dropped bytes before, so fail loudly.
    assert len(input_ids) == len(boundaries), (
        f"input_ids ({len(input_ids)}) and boundaries ({len(boundaries)}) must align 1:1"
    )

    # Use the literal '|' byte as the visual separator. ByT5 appends eos, so
    # add_special_tokens=False is required or [-1] would grab eos, not '|'.
    separator = tokenizer("|", add_special_tokens=False)["input_ids"][-1]

    # Group bytes into patches. The model's convention (see shortening.downsample
    # / downsample_mean) is "a 1 closes the patch at t; the split happens after t",
    # so the separator goes *after* the boundary byte.
    tokens = []
    current_token = []
    for token_id, is_boundary in zip(input_ids, boundaries):
        current_token.append(int(token_id))
        if is_boundary == 1:
            current_token.append(separator)
            tokens.extend(current_token)
            current_token = []
    if current_token:                       # flush trailing (unterminated) patch
        current_token.append(separator)
        tokens.extend(current_token)

    decoded = tokenizer.decode(tokens, skip_special_tokens=True).strip("|")
    print(f"Patches: {decoded.count('|') + 1}  (boundaries: {int(boundaries.sum())})")
    print(decoded)
    return decoded


def generate(model, tokenizer, prompt, args, device, drafter=None):
    """Run one of the four decoding paths through model.generate()."""
    mode_kwargs = MODES[args.mode]
    tokenized_text = tokenizer(prompt, add_special_tokens=False)
    input_ids = torch.tensor(tokenized_text["input_ids"], device=device).unsqueeze(0)

    with torch.no_grad():
        out, acceptance_rate = model.generate(
            input_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_token_id=tokenizer.eos_token_id,
            repetition_penalty=args.repetition_penalty,
            threshold=args.threshold,
            candidates=args.candidates,
            drafter=drafter,
            **mode_kwargs,
        )

    output = tokenizer.decode(out, skip_special_tokens=False)
    print(f"[{args.mode}] acceptance rate: {acceptance_rate:.4f}")
    print(output)
    return output


def main():
    args = pargs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt = args.prompt
    if prompt is None:
        with open(args.prompt_file) as f:
            prompt = f.read()

    model, tokenizer, model_config = load_model_and_tokenizer(args.model_path, args, device)
    print(f"Loaded model from {args.model_path}")

    drafter = None
    if args.mode == "drafter":
        drafter, _, _ = load_model_and_tokenizer(args.drafter_path, args, device)
        print(f"Loaded drafter from {args.drafter_path}")

    if args.show_tokenization:
        raw = model_config["id_to_script"]
        # Support both formats:
        #   old: {"259": "<en>", ...}  (int-string -> script-name)
        #   new: {"<en>": 259, ...}    (script-name -> int)
        if all(str(k).lstrip("-").isdigit() for k in raw.keys()):
            script_to_id = {v: int(k) for k, v in raw.items()}
        else:
            script_to_id = {k: int(v) for k, v in raw.items()}
        tokenize(model, tokenizer, prompt, lang_token_id=script_to_id["<en>"], device=device)

    generate(model, tokenizer, prompt, args, device, drafter=drafter)


if __name__ == "__main__":
    main()
