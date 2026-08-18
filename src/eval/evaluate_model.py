#!/usr/bin/env python3
# Evaluation script to a pretrained FlexiTokens model
#run with:
# python src/eval/evaluate_model.py --model_path /path/to/model --output_dir /path/to/output --eval_split test
import os
import argparse
import torch
import json
from datetime import datetime
from accelerate.logging import get_logger
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator, DistributedDataParallelKwargs

from src.utils.data_utils import FxTDataset
from src.eval.evaluation import evaluate_inidiv_dataset_LM
from src.utils.utils import init_seed
from src.eval.model_loader import load_fxt_model
import warnings

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained FlexiTokens model')
    
    parser.add_argument('--model_path', type=str, default=None, required=True,
                        help='Path to the model directory containing checkpoint and config.json')
    parser.add_argument('--checkpoint_name', type=str, default='model.pth',
                        help='Name of the checkpoint file within the model directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--eval_split', type=str, default='test',
                        choices=['test', 'validation'],
                        help='Split to evaluate on (test or validation)')
    parser.add_argument('--tokenizer_path', type=str, default='google/byt5-small',
                        help='Path to the tokenizer')

    
    return parser.parse_args()


def main():
    args = parse_args()
    args.cache_dir = "/path/to/hf_cache"
    args.load_from_disk = True  # Always load from disk for evaluation
    # Load configuration from the model's config.json
    config_path = os.path.join(args.model_path, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    # Set up accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    logger.info(accelerator.state, main_process_only=False)

    # Set seed
    init_seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer using model_loader
    model, tokenizer, model_config = load_fxt_model(args.model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    # Prepare id to script mapping (as in train.py)
    id_to_script = {value: key for key, value in model_config["id_to_script"].items()}
    language_to_script_id = {lang: int(id_to_script[script]) for lang, script in model_config["language_to_script"].items()}
    logger.info(f"language_to_script_id is {language_to_script_id}")

    # Set up dataset (as in train.py)
    boundary_kwargs = {
        'boundaries_type': model_config.get('boundaries_type', None),
        'fixed_sf': model_config.get('fixed_sf', None),
        'tokenizer_path': args.tokenizer_path or args.model_path,
        'script_tokens': model_config["script_tokens"],
        'cache_dir':args.cache_dir,
    }
    ## config to args
    for k, v in model_config.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    data_corpus = FxTDataset(
        model_config["data"], model_config["seq_len"], accelerator, language_to_script_id, args=args, **boundary_kwargs
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

    # Prepare model with accelerator
    model = accelerator.prepare(model)

    # Evaluate individual languages (BPB) on test or validation set
    logger.info("Starting evaluation on individual languages (BPB)")
    split = args.eval_split
    eval_dataset = data_corpus.individual_test_dataset if split == "test" else data_corpus.individual_validation_dataset

    languages_bpc_dictionary, _ = evaluate_inidiv_dataset_LM(
        eval_dataset,
        data_collator,
        args.eval_batch_size,
        accelerator,
        model,
        task="EVAL" if "bytes" in args.model_path else "LM"
        )

    # Save results and model name to the file name.
    model_name = os.path.basename(args.model_path.rstrip("/"))
    results_path = os.path.join(args.output_dir, f"{model_name}_language_{split}_eval_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    logger.info(f"Saving evaluation results to {results_path}")

    # Only write results on main process
    if accelerator.is_main_process:
        with open(results_path, 'w') as f:
            json.dump(languages_bpc_dictionary, f, indent=4)

    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()