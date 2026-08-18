import os
os.environ["NCCL_TIMEOUT"] = "540000000" 
import argparse
import math
import numpy as np
import torch
import torch.optim as optim
import yaml
import json
import logging
import transformers
from datetime import datetime
from collections import defaultdict
from accelerate.logging import get_logger
from transformers import DataCollatorForLanguageModeling, get_scheduler, set_seed
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets.distributed import split_dataset_by_node
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
from src.utils.data_utils import FxTDataset, MixtureByteVocab
from src.eval.evaluation import evaluate_inidiv_dataset_LM
from src.finetune.data_utils import dynamic_padding_data_collator, JointInputcorpus

from src.model.modern_fxt import FxTTransformerLM


from src.utils.utils import (
    init_seed,
    calculate_mean,
    save_args_to_json,
    count_trainable_parameters,
    get_model_config,
    grad_norm,
    block_grad_norms,
    save_clean_model_weights,
)
import warnings

warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)
logger = get_logger(__name__)


def list_of_strings(arg):
    return arg.split(",")


def list_of_ints(arg):
    return list(map(float, arg.split(",")))


def parse_args():
    parent_parser = argparse.ArgumentParser(add_help=False)

    parser = argparse.ArgumentParser(parents=[parent_parser])
    cfg_parser = argparse.ArgumentParser(parents=[parent_parser])
    cfg_parser.add_argument("--config", default="default")
    cfg_parser.add_argument("--config_file", default=None)

    config_args, _ = cfg_parser.parse_known_args()

    assert config_args.config is not None and config_args.config_file is not None
    with open(config_args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[config_args.config]["train"]

    # Main args
    general = parser.add_argument_group("general setup")
    general.add_argument(
        "--config_file",
        default=config_args.config_file,
        type=str,
        help="Path to the config file",
    )
    general.add_argument(
        "--exp_dir", default=None, type=str, help="Directory for all the experiments"
    )
    general.add_argument(
        "--work_dir",
        default=None,
        type=str,
        help="Directory for the results. This is used when --exp_dir is not set",
    )

    dataset = parser.add_argument_group("dataset setup")
    dataset.add_argument("--data", type=str, help="Location of the data corpus")
    ## stream
    # cache
    dataset.add_argument(
        "--cache_dir", type=str, help="Location of the cache directory"
    )
    dataset.add_argument(
        "--num_proc", type=int, help="Number of processes to use for tokenization"
    )
    dataset.add_argument(
        "--load_from_disk",
        action="store_true",
        help="Load dataset from disk",
    )
    dataset.add_argument("--language_to_script", help="Language to script mapping")
    dataset.add_argument("--train_size", help="Size of the training set")
    dataset.add_argument("--val_size", help="Size of the validation set")

    model = parser.add_argument_group("model setup")
    #TODO: first 5 no longer needed, please remove    
    model.add_argument("--dropout", type=float, default=0.1, help="Global dropout rate")
    model.add_argument(
        "--model_config",
        type=str,
        default="[2, (10,) ,2, 2]",
        help="[pre_layers, (shortened_layers, ), post_layers, mtp_layers]",
    )
    model.add_argument("--activation_function", type=str, default="relu")
    model.add_argument("--num_predictors", type=int, default=3)
    model.add_argument("--use_binomial", type=bool, default=False)
    model.add_argument(
        "--s_lower_bound",
        type=float,
        default=2.0,
        help="Lower bound scaling factor for sigmoid",
    )
    model.add_argument(
        "--scale_bp",
        type=float,
        default=1.0,
        help="Scaling factor for boundary prediction loss",
    )
    model.add_argument(
        "--scale_loss2",
        type=float,
        default=0.6,
        help="Scaling factor for the second loss (e.g. group attention loss)",
    )
    model.add_argument(
        "--lm_tie_weights",
        type=bool,
        default=False,
        help="Whether to tie the weights of the language model",
    )
    model.add_argument(
        "--lm_rms_eps",
        type=float,
        default=1e-6,
        help="Epsilon for RMS normalization in the language model",
    )
    model.add_argument(
        "--use_group_attn",
        type=bool,
        default=False,
        help="Whether to use group attention",
    )
    model.add_argument(
        "--attn_type",
        type=str,
        default="previous",
        choices=["previous", "previous_self", "previous_same"],
        help="Type of attention to use",
    )

    boundaries = parser.add_argument_group("boundary creator")
    boundaries.add_argument("--boundaries_type", type=str)
    boundaries.add_argument(
        "--tokenizer_path",
        type=str,
        default="google/byt5-small",
        help="Path to the tokenizer",
    )
    boundaries.add_argument("--fixed_sf", type=int)
    boundaries.add_argument("--spikes_left", type=int)
    boundaries.add_argument("--temp", type=float)
    boundaries.add_argument("--script_tokens", type=list_of_strings)
    boundaries.add_argument("--prior_list", type=list_of_ints)
    boundaries.add_argument("--prior_std", type=list_of_ints)

    opt = parser.add_argument_group("optimizer setup")
    opt.add_argument(
        "--optim", default="adam", type=str, choices=["adam"], help="Optimizer to use"
    )
    opt.add_argument("--lr", type=float, default=0.00025, help="Initial learning rate")
    opt.add_argument(
        "--scheduler",
        default="cosine",
        type=str,
        choices=["cosine"],
        help="LR scheduler to use",
    )
    opt.add_argument("--clip", type=float, default=0.25, help="Gradient clipping")
    opt.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay for adam"
    )
    opt.add_argument("--adam_b1", type=float, default=0.9)
    opt.add_argument("--adam_b2", type=float, default=0.999)
    opt.add_argument("--adam_eps", type=float, default=1e-8)

    training = parser.add_argument_group("training setup")
    training.add_argument(
        "--task_type", type=str, default="LM", help="Type of task: LM or SFT"
    )
    training.add_argument(
        "--max_train_steps", type=int, default=None, help="Max number of training steps"
    )
    training.add_argument(
        "--batch_size", type=int, default=64, help="Global batch size"
    )
    training.add_argument("--seed", type=int, default=42, help="Random seed")
    training.add_argument(
        "--seq_len", type=int, default=512, help="Maximum sequence length"
    )
    training.add_argument("--report_to", type=str, default="wandb", help="Wandb or trackio")
    training.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    training.add_argument(
        "--num_warmup_steps", type=int, default=5000, help="Number of warm up steps"
    )
    training.add_argument(
        "--logging_steps", type=int, default=500, help="Number of logging steps"
    )
    training.add_argument(
        "--checkpointing_steps", type=str, help="Checkpointing steps", default="5000"
    )
    training.add_argument(
        "--with_tracking", type=bool, help="whether to track with wandb ?", default=False
    )
    training.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Whether to resume training from a checkpoint",
        default=False,
    )
    training.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs.",
    )
    training.add_argument(
        "--log_dir", type=str, default="./wandb_logs", help="Log directory"
    )
    training.add_argument(
        "--freeze_pretrained",
        action="store_true",
        help="Freeze the pretrained backbone and train only the MTP heads",
    )

    val = parser.add_argument_group("validation setup")
    val.add_argument("--ckpt_path", type=str, default="")
    val.add_argument("--eval_batch_size", type=int, default=64)

    parser.set_defaults(**config)
    args, _ = parser.parse_known_args()

    args.ckpt_path = "/".join(config_args.config_file.split("/")[:-1])

    assert args.boundaries_type in [
        "none",
        "fixed",
        "whitespaces",
        "unigram",
        "gumbel",
        "hnet",
    ]

    return args


def evaluate(model, eval_dataloader, accelerator, batch_size, step=None, phase="eval", task="LM", script_id=None):
    """Evaluation function to be called both during training and at the end of epochs"""
    print("Starting evaluation...")
    accelerator.wait_for_everyone()
    # Unwrap DDP to avoid inter-rank collectives during eval
    losses, losses2, val_aux_losses = [], [], []
    val_stats_agg = defaultdict(list)

    for eval_step, batch in enumerate(eval_dataloader):
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        with torch.no_grad():
            loss, loss2, val_stats, val_aux_loss, _ = model(batch, task=task, script_id=script_id)

        losses.append(loss.detach())
        losses2.append(loss2.detach())
        if len(val_aux_loss) > 1:
            val_aux_losses.append(
                torch.mean(torch.stack(val_aux_loss)).detach()
            )
        else:
            val_aux_losses.append(val_aux_loss[0].detach())

        for k, v in val_stats.items():
            val_stats_agg[f"{phase}_{k}"].append(v)
    val_stats_mean_dict = calculate_mean(val_stats_agg)

    losses = torch.stack(losses)
    losses2 = torch.stack(losses2)
    val_aux_losses = torch.stack(val_aux_losses)


    eval_loss = torch.mean(losses)
    eval_loss2 = torch.mean(losses2)
    eval_val_aux_loss = torch.mean(val_aux_losses)
    eval_bpc = eval_loss / math.log(2)
    eval_bpc2 = eval_loss2 / math.log(2)
 

    # Create metrics dictionary
    eval_metrics = {
        f"{phase}_bpc": eval_bpc.item(),
        f"{phase}_bpc2": eval_bpc2.item(),
        f"{phase}_lm_loss": eval_loss.item(),
        f"{phase}_lm_loss2": eval_loss2.item(),
        f"{phase}_aux_loss": eval_val_aux_loss.item(),
    }

    # Update with validation stats means
    eval_metrics.update(val_stats_mean_dict)
    step_info = f" at step {step}" if step is not None else ""
    logger.info(
        f"Evaluation{step_info}: {phase}_bpc: {eval_bpc} {phase}_lm_loss: {eval_loss}; {phase}_bpc2: {eval_bpc2}, {phase}_lm_loss2: {eval_loss2} \n {phase}_aux_loss: {eval_val_aux_loss}"
    )

    logger.info(val_stats_mean_dict)

    return eval_metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    # Create output directory with timestamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.work_dir is None:
        config_file_name = os.path.basename(args.config_file)
        config_file_name = os.path.splitext(config_file_name)[0]
        args.work_dir = os.path.join(args.exp_dir, config_file_name, current_time)
    basename = f"{os.path.basename(args.work_dir)}_{current_time}"
    new_path = os.path.join(os.path.dirname(args.work_dir), basename)
    args.output_dir = new_path
    os.makedirs(args.output_dir, exist_ok=True)

    # Accelerate config
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["project_dir"] = args.output_dir
    kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True
    )  # please set to true if you have unused parameters
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # kwargs_handlers=[kwargs],
        **accelerator_log_kwargs,

    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    init_seed(args.seed)

    ###########################################################################
    # Load data
    ###########################################################################
    ## add eot_token to the script tokens
    args.script_tokens = ["<en>", "<eot>"]
    boundary_kwargs = {
        "boundaries_type": args.boundaries_type,
        "fixed_sf": args.fixed_sf,
        "tokenizer_path": args.tokenizer_path,
        "script_tokens": args.script_tokens,
        "cache_dir": args.cache_dir,
    }
    logging.info("----------------------------------------------------------------")
    logging.info("----------------------------------------------------------------")
    logging.info("Preparing corpus")

    vocab = MixtureByteVocab(**boundary_kwargs)

    if len(vocab) > 1000:
  
        args.id_to_script = {
            len(vocab): "<en>", len(vocab) + 1: "<eot>"
        }

    else:
        args.id_to_script = {
        vocab.tokenizer.convert_tokens_to_ids(i): i  for i in args.script_tokens
    }

    
   
    script_id = list(args.id_to_script.keys())[0]
    language_to_script_id = {value: key for key, value in args.id_to_script.items() if value in list(args.language_to_script.values())}
    args.all_script_ids_dict = {
        j: i for i, j in zip(zip(args.prior_list, args.prior_std), args.script_tokens)
    }  # the prior list is the prior probability of each script
    print(f"after language_to_script_id is {language_to_script_id}")

    # lm_vocab_size here determines the model embedding size
    args.lm_vocab_size = args.n_token = len(vocab)
    args.ignore_index = -100 
    args.mtp_layers = eval(args.model_config)[-1]
    print(args.ignore_index, "ignore index", args.tokenizer_path)
    logging.info(f"Script ids dict is  {args.all_script_ids_dict}")

    # Save config file
    save_args_to_json(args, args.output_dir)
    vocab.tokenizer.save_pretrained(args.output_dir)

    if args.task_type == "SFT":
        logger.info("Loading SFT corpus ....")
        data_corpus = JointInputcorpus(
            language="en",
            dataset_name=args.dataset_name,
            tokenizer=vocab.tokenizer,
            max_seq_length=args.seq_len,
            accelerator=accelerator,
            cache_dir=args.cache_dir,
            language_to_script_id=language_to_script_id,
            args=args,
        )

        data_collator = lambda x: dynamic_padding_data_collator(
            x, vocab.tokenizer
        )
        data_collator.tokenizer = vocab.tokenizer 
        train_dataset_to_shard = data_corpus.train_dataset.shuffle(
            seed=args.seed, buffer_size=10000
        )
        # Shard the dataset per GPU rank for distributed training
        train_dataset_sharded = split_dataset_by_node(
            train_dataset_to_shard,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
        train_dataloader = StatefulDataLoader(
            train_dataset_sharded,
            collate_fn=data_collator,
            batch_size=args.batch_size,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=args.num_proc,
            persistent_workers=args.num_proc > 0,
        )
        initial_train_dl_state = train_dataloader.state_dict()


        eval_dataloader = StatefulDataLoader(
            data_corpus.validation_dataset,
            collate_fn=data_collator,
            batch_size=args.batch_size,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=args.num_proc,
            persistent_workers=args.num_proc > 0,
        )

    else:

        # Load dataset
        data_corpus = FxTDataset(
            args.data,
            args.seq_len,
            accelerator,
            language_to_script_id,
            args,
            **boundary_kwargs,
        )
        # Dataloader and data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=vocab.tokenizer, mlm=False, return_tensors="pt"
        )
        # Enable per-epoch shuffling before sharding — `.shuffle` on an IterableDataset
        # installs a rolling buffer; `set_epoch(epoch)` (called below in the loop) reseeds
        # the buffer each epoch so examples land in a fresh order every pass.
        train_dataset_to_shard = data_corpus.train_dataset.shuffle(
            seed=args.seed, buffer_size=10000
        )
        # Shard the dataset per GPU rank for distributed training
        train_dataset_sharded = split_dataset_by_node(
            train_dataset_to_shard,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )

        train_dataloader = StatefulDataLoader(
            train_dataset_sharded,
            collate_fn=data_collator,
            batch_size=args.batch_size,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=args.num_proc,
            persistent_workers=args.num_proc > 0,
            drop_last=True,
        )
        # Snapshot of the fresh (pre-iteration) loader state. We restore this between
        # outer epochs so each new epoch starts at position 0 of the shard; without
        # this, StatefulDataLoader persists its position across iter() calls and every
        # epoch after the first re-exhausts at the same point the previous one did.
        initial_train_dl_state = train_dataloader.state_dict()

        eval_dataloader = StatefulDataLoader(
            data_corpus.validation_dataset,
            collate_fn=data_collator,
            batch_size=args.eval_batch_size,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=args.num_proc,
            persistent_workers=args.num_proc > 0,
            drop_last=True,

        )

    ###########################################################################
    # Build model
    ###########################################################################

    output_dir = args.output_dir.lower()
    blocks = eval(args.model_config)
    main_decoder = blocks[2]
    aux_decoder = blocks[3]

    model_config = get_model_config(args, FxTTransformerLM)
    print(model_config)
    logger.info(f"Main decoder has {main_decoder} layers")
    logger.info(f"MTP decoder has {aux_decoder} layers")
    model = FxTTransformerLM(args)
    
    if getattr(args, "freeze_pretrained", False) and hasattr(model, "freeze_pretrained"):
        model.freeze_backbone()
        main_decoder = 0
        logger.info("Pretrained backbone frozen; training MTP heads only")
    else:
        logger.info("Training the entire model")
        model.unfreeze_backbone()

    
    if not args.resume_from_checkpoint:
        torch._dynamo.config.optimize_ddp = False
        model = torch.compile(model, dynamic=True)
        print("Model compiled with torch.compile")
        pass
    

    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    model.tokenizer = vocab.tokenizer
    logger.info(model)

    n_params = count_trainable_parameters(model)
    logger.info(
        f"Training new model from scratch - Total size={n_params/1000000:.2f}M params"
    )


    init_optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_b1, args.adam_b2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    overrode_max_train_steps = False
    init_scheduler = get_scheduler(
        name="cosine",
        optimizer=init_optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes
        ),
    )

    model, optimizer, lr_scheduler = (
        accelerator.prepare(
            model, init_optimizer, init_scheduler
        )
    )
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    

    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "scheduler"
        ]  # .value

    total_batch_size = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    ###########################################################################
    # Start Training
    ###########################################################################

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.with_tracking:
        accelerator.init_trackers(
            project_name="fxt-mtp",
            config=experiment_config,
            init_kwargs={"wandb": {"entity": os.environ.get("WANDB_ENTITY"), "name": basename, "dir": args.log_dir}},
        )

        # pass
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0
    best_eval_loss = float("inf")
    resume_step = None
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch  # * num_update_steps_per_epoch
        elif "step" in training_difference:
            resume_step = int(training_difference.replace("step_", ""))
            completed_steps = resume_step
        else:
            resume_step = 0
            completed_steps = 0
        if completed_steps == 0:
            #load only the models weights but not the optimizer and scheduler states
            accelerator.print(f"Resuming from model weights at checkpoint: {checkpoint_path}")
            from safetensors.torch import load_model
            load_model(
                accelerator.unwrap_model(model),
                os.path.join(checkpoint_path, "model.safetensors"),
                device=str(accelerator.device),
            )
        else:
            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            
            # Restore StatefulDataLoader position — O(1), no sample iteration needed
            dl_state_path = os.path.join(
                checkpoint_path, f"dataloader_state_{accelerator.process_index}.pt"
            )
            if os.path.exists(dl_state_path):
                train_dataloader.load_state_dict(torch.load(dl_state_path, weights_only=True))
                logger.info(f"Restored dataloader state from {dl_state_path}")
            else:
                logger.warning(
                    f"No dataloader state found at {dl_state_path}; "
                    "training will restart from the beginning of the dataset."
                )
        torch._dynamo.config.optimize_ddp = False
        model = torch.compile(model, dynamic=True)

        
    metrics_dict = {}
    current_scale_loss2 = 0
    print(f"Resuming training from epoch {starting_epoch} and step {completed_steps}")    
    progress_bar.update(completed_steps)
    for epoch in range(starting_epoch, args.num_train_epochs):

        if epoch > starting_epoch:
            train_dataloader.load_state_dict(initial_train_dl_state)
            # Reseed the shuffle buffer with the epoch number (deterministic across runs).
            train_dataset_sharded.set_epoch(epoch)
            print(f"Starting epoch {epoch+1} after completing {completed_steps} steps.")

        eval_metrics = evaluate(
            model, eval_dataloader, accelerator, args.eval_batch_size
        )
        total_train_loss, total_train_loss2, total_train_aux_loss, total_count = torch.tensor(0.0, device=accelerator.device), torch.tensor(0.0, device=accelerator.device), torch.tensor(0.0, device=accelerator.device), 0

        dist_active = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and accelerator.num_processes > 1
        )
        def _synced_epoch(loader):
            it = iter(loader)
            while True:
                try:
                    batch = next(it)
                    local_done = 0
                except StopIteration:
                    batch = None
                    local_done = 1
                if dist_active:
                    flag = torch.tensor([local_done], device=accelerator.device)
                    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
                    done = flag.item() > 0
                else:
                    done = local_done > 0
                if done:
                    return
                yield batch
        active_dataloader = _synced_epoch(train_dataloader) #if args.task_type == "LM" else train_dataloader
        train_stats_agg = defaultdict(list)
        # t0 = time.time()
        for step, batch in enumerate(
            tqdm(active_dataloader, disable=not accelerator.is_local_main_process)
        ):
            # t1 = time.time()
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}

            with accelerator.accumulate(model):
                seq_loss, seq_loss2, stats, aux_loss, logits = model(batch, task=args.task_type, script_id=script_id)
                if main_decoder == 0 and aux_decoder > 0:
                    loss = seq_loss2 + args.scale_bp * aux_loss[0]
                elif main_decoder > 0 and aux_decoder > 0:
                    #the main mtp loss
                    current_scale_loss2 = args.scale_loss2
                    loss = seq_loss + (args.scale_bp * aux_loss[0]) + (current_scale_loss2 * seq_loss2)
                elif main_decoder > 0 and aux_decoder == 0:
                    loss = seq_loss + args.scale_bp * aux_loss[0]
                elif main_decoder == 0 and aux_decoder == 0:
                    loss = seq_loss
        
                total_train_loss += seq_loss.detach().float()
                total_train_loss2 += seq_loss2.detach().float() 
                if len(aux_loss) > 1:
                    total_train_aux_loss += (
                        torch.mean(torch.stack(aux_loss)).detach().float()
                    )
                else:
                    total_train_aux_loss += aux_loss[0].detach().float()
                total_count += 1

                for k, v in stats.items():
                    train_stats_agg[f"train_{k}"].append(v)

                current_lr = lr_scheduler.get_last_lr()[0]
                accelerator.backward(loss)

                # Apply gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if step % args.logging_steps == 0:
                if args.with_tracking:

                    grad_norm_ = grad_norm(model)
                    block_grad_norms_ = block_grad_norms(model)
                    accelerator.log(
                        {
                            "train_lm_loss": seq_loss.item(),
                            "train_lm_loss2": seq_loss2.item(),
                            "train_bpc1": seq_loss.item() / math.log(2),
                            "train_bpc2": seq_loss2.item() / math.log(2),
                            "current_lr": current_lr,
                            "grad_norm": grad_norm_,
                            "current_scale_loss2": current_scale_loss2,
                            **block_grad_norms_,
                        },
                        step=completed_steps,
                    )
                    accelerator.log(
                        stats,
                        step=completed_steps,
                    )
            
            # t2 = time.time()
            # logger.info(f"Time taken for step {step} in epoch {epoch} is {t2-t1} seconds. Data loading time is {t1-t0} seconds.")
            # t0 = time.time()
            torch.cuda.synchronize()
            accelerator.wait_for_everyone()
            if isinstance(checkpointing_steps, int):
                if (completed_steps % checkpointing_steps == 0 and completed_steps > 0):
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    save_clean_model_weights(accelerator, model, output_dir)
                    torch.save(
                        train_dataloader.state_dict(),
                        os.path.join(output_dir, f"dataloader_state_{accelerator.process_index}.pt"),
                    )

            if completed_steps % args.eval_interval == 0 and completed_steps > 0:
                
                print("Evaluating the model at step ", completed_steps)
                eval_metrics = evaluate(
                    model,
                    eval_dataloader,
                    accelerator,
                    args.eval_batch_size,
                    completed_steps,
                    script_id=script_id
                )
                accelerator.wait_for_everyone()
                # Log the evaluation metrics
                if args.with_tracking:
                    accelerator.log(eval_metrics, step=completed_steps)

                # Save model if it's the best so far
                if eval_metrics["eval_lm_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_lm_loss"]
                    output_dir = "best_model"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    save_clean_model_weights(accelerator, model, output_dir)
                    torch.save(
                        train_dataloader.state_dict(),
                        os.path.join(output_dir, f"dataloader_state_{accelerator.process_index}.pt"),
                    )
            

            if completed_steps >= args.max_train_steps:
                break

        # take the mean afer every epoch
        train_stats_mean_dict = calculate_mean(train_stats_agg)

        
        ###########################################################################
        # Evaluate Validation Set after each epoch
        ###########################################################################
        if completed_steps >= args.max_train_steps or epoch == args.num_train_epochs - 1:
            print("Saving final model...")
            final_output_dir = args.output_dir
            accelerator.save_state(final_output_dir)
            save_clean_model_weights(accelerator, model, final_output_dir)
            torch.save(
                train_dataloader.state_dict(),
                os.path.join(final_output_dir, f"dataloader_state_{accelerator.process_index}.pt"),
            )

        torch.cuda.synchronize()
        accelerator.wait_for_everyone()
        eval_metrics = evaluate(
            model,
            eval_dataloader,
            accelerator,
            args.eval_batch_size,
            completed_steps,
            script_id=script_id
        )
        accelerator.wait_for_everyone()

        train_loss = total_train_loss.item() / total_count
        train_loss2 = total_train_loss2.item() / total_count
        train_aux_loss = total_train_aux_loss.item() / total_count

        eval_loss = eval_metrics["eval_lm_loss"]
        eval_loss2 = eval_metrics["eval_lm_loss2"]
        eval_bpc = eval_metrics["eval_bpc"]
        eval_val_aux_loss = eval_metrics["eval_aux_loss"]

        logger.info(
            f"epoch {epoch}: train_bpc: {train_loss / math.log(2)}  train_loss: {train_loss}   train_loss2: {train_loss2}  train_aux_loss: {train_aux_loss} eval_bpc: {eval_bpc} eval_loss: {eval_loss}  eval_aux_loss: {eval_val_aux_loss} "
        )

        metrics_dict = {
            "train_bpc": train_loss / math.log(2),
            "train_lm_loss": train_loss,
            "train_lm_loss2": train_loss2,
            "train_aux_loss": train_aux_loss,
            "epoch": epoch,
            "step": completed_steps,
        }

        # Update with evaluation metrics
        metrics_dict.update(eval_metrics)
        metrics_dict.update(train_stats_mean_dict)
        print("\n\n\n\n\n\n",metrics_dict)

        if args.with_tracking:
            accelerator.log(
                metrics_dict,
                step=completed_steps,
            )


    if args.output_dir is not None:

        accelerator.save_state(output_dir)
        save_clean_model_weights(accelerator, model, output_dir)
        torch.save(
            train_dataloader.state_dict(),
            os.path.join(output_dir, f"dataloader_state_{accelerator.process_index}.pt"),
        )


    # Evaluate individual languages
    logger.info("Start evaluating individual languages test set...")
    
    languages_bpc_dictionary, languages_loss_dictionary = evaluate_inidiv_dataset_LM(
        data_corpus.individual_test_dataset,
        data_collator,
        args.batch_size,
        accelerator,
        model,
    )

    languages_bpc_dictionary.update(languages_loss_dictionary)
    accelerator.wait_for_everyone()


    # log languages dict
    if args.with_tracking:
        accelerator.log(
            languages_bpc_dictionary,
            step=completed_steps,
        )

    if args.output_dir is not None and accelerator.is_main_process:
        with open(
            os.path.join(args.output_dir, f"all_results_{str(step)}.json"), "w"
        ) as f:
            json.dump(metrics_dict, f)

        with open(
            os.path.join(
                args.output_dir, f"language_eval_results_{str(step)}.json"
            ),
            "w",
        ) as f:
            json.dump(languages_bpc_dictionary, f)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    logger.info("Training completed.")


if __name__ == "__main__":
    main()