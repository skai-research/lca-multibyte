import json
import os
import shutil
import numpy as np
import torch
import random


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_mean_with_padding(batch_tensor, batch_masks):
    expanded_masks_batch = batch_masks.unsqueeze(-1).expand_as(batch_tensor)
    masked_tensor = batch_tensor * expanded_masks_batch
    sum_tensor = masked_tensor.sum(dim=1)
    count_tensor = (expanded_masks_batch != 0).sum(dim=1)
    mean_tensor = sum_tensor / count_tensor

    return mean_tensor


def read_json_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        return

    os.makedirs(dir_path, exist_ok=True)

    print("Experiment dir : {}".format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, "scripts")
        os.makedirs(script_path, exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_args_to_json(args, folder_path):
    args_dict = vars(args)
    with open(os.path.join(folder_path, "config.json"), "w") as json_file:
        json.dump(args_dict, json_file, indent=4)

    print("Arguments saved to {}".format(os.path.join(folder_path, "config.json")))


def load_checkpoint(path):
    if os.path.isdir(path):
        path = os.path.join(path, "checkpoint_last.pt")

    dst = f"cuda:{torch.cuda.current_device()}"
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=dst)
    return checkpoint


def save_ckpt(model, optimizer, path, epoch):
    torch.save(model, os.path.join(path, "model_{}.pt".format(epoch)))
    torch.save(
        optimizer.state_dict(), os.path.join(path, "optimizer_{}.pt".format(epoch))
    )


def calculate_mean(data_dict):
    """
    Calculate the mean for each key in a defaultdict.
    """
    mean_dict = {}
    for key, values in data_dict.items():
        if isinstance(
            values[0], torch.Tensor
        ):  # Check if the first value is a PyTorch tensor
            mean_dict[key] = torch.stack(values).mean(dim=0).item()
        else:
            mean_dict[key] = sum(values) / len(values)
    return mean_dict


def get_model_config(config, model_class):
    import inspect

    model_args = inspect.getfullargspec(model_class).args
    assert model_args.index("self") == 0
    model_args = model_args[1:]
    try:
        values = {arg: getattr(config, arg) for arg in model_args}
    except:
        values = {arg: config[arg] for arg in model_args if arg in config}
    return values


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def _module_grad_norm(module, n_layers=1):
    """Compute L2 gradient norm for a single nn.Module, normalized by layer count.

    When *n_layers* > 1 the raw L2 norm is divided by n_layers so that
    blocks with different depths are comparable.
    """
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return (total ** 0.5) / 1


def block_grad_norms(model):
    """Return a dict mapping block names to their L2 gradient norms.

    Decoder-stack norms are **normalized by layer count** so that blocks
    with different depths (e.g. 2 pre-layers vs 16 shortened layers) are
    directly comparable.

    Works with any FxTTransformerLM-style model that has:
      - blocks: nn.ModuleList of nn.ModuleLists
            blocks[0] = pre_layers
            blocks[1] = shortened_layers
            blocks[2] = post_layers   (main decoder)
            blocks[3] = mtp_layers    (aux decoder)
      - token_embedding / word_emb
      - norm
      - head / final_cast
      - script_to_bp_layers  (boundary predictor, optional)

    Returns:
        dict[str, float]: e.g. {"grad_norm/pre_layers": 0.12, ...}
    """
    BLOCK_NAMES = ["pre_layers", "shortened_layers", "post_layers", "mtp_layers"]
    norms = {}

    # --- Unwrap DDP / FSDP if needed ---
    raw = model.module if hasattr(model, "module") else model

    # --- Decoder block stacks (normalized by number of layers) ---
    if hasattr(raw, "blocks"):
        for idx, name in enumerate(BLOCK_NAMES):
            if idx < len(raw.blocks) and len(raw.blocks[idx]) > 0:
                n_layers = len(raw.blocks[idx])
                norms[f"grad_norm/{name}"] = _module_grad_norm(
                    raw.blocks[idx], n_layers=n_layers
                )

    # --- Embedding ---
    for attr in ("token_embedding", "word_emb"):
        if hasattr(raw, attr):
            norms["grad_norm/embedding"] = _module_grad_norm(getattr(raw, attr))
            break

    # --- LM head ---
    for attr in ("head", "final_cast"):
        if hasattr(raw, attr):
            norms["grad_norm/lm_head"] = _module_grad_norm(getattr(raw, attr))
            break

    # --- Boundary predictor ---
    if hasattr(raw, "script_to_bp_layers"):
        norms["grad_norm/boundary_predictor"] = _module_grad_norm(raw.script_to_bp_layers)

    # --- Final norm ---
    if hasattr(raw, "norm"):
        norms["grad_norm/final_norm"] = _module_grad_norm(raw.norm)

    return norms


def linear_anneal(current_step, max_steps, max_value, warmup_fraction=0.1):
    """Linearly anneal a value from 0 to max_value over the first
    `warmup_fraction` of training, then hold at max_value.

    Args:
        current_step (int): Current training step.
        max_steps (int): Total number of training steps.
        max_value (float): Target value to anneal towards.
        warmup_fraction (float): Fraction of max_steps over which to
            ramp from 0 to max_value (default 0.1 = 10%).

    Returns:
        float: The annealed value.
    """
    warmup_steps = int(max_steps * warmup_fraction)
    if warmup_steps == 0:
        return max_value
    if current_step >= warmup_steps:
        return max_value
    return max_value * (current_step / warmup_steps)


def save_clean_model_weights(accelerator, model, output_dir):
    """Save model weights without _orig_mod. prefix added by torch.compile."""
    from safetensors.torch import save_model
    unwrapped = getattr(model, "module", model)       # DDP
    unwrapped = getattr(unwrapped, "_orig_mod", unwrapped)  # torch.compile
    # save_model handles tied weights (e.g. lm_head / embed_tokens sharing memory)
    # by deduplicating them — save_file would raise RuntimeError on shared tensors.
    save_model(unwrapped, os.path.join(output_dir, "model.safetensors"))
    print(f"Model weights saved to {os.path.join(output_dir, 'model.safetensors')}")
