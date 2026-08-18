import argparse
import json
import logging
import math
import os
import transformers
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
)

from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datetime import datetime
from transformers import get_scheduler
from transformers import AutoTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader

from safetensors.torch import load_file
from src.model.modern_fxt import FxTTransformerLM
from src.utils.utils import (
    read_json_file,
    save_args_to_json,
    count_trainable_parameters,
    grad_norm,
    block_grad_norms,
    save_clean_model_weights,
)
from src.finetune.data_utils import dynamic_padding_data_collator, JointInputcorpus
from src.train.train import evaluate  as evaluate_model #using this because of the evaluate package.


logger = get_logger(__name__)


def load_pretrained_model(args):
    pretrained_path = args.pretrained_path.lower()
    args.model_path = pretrained_path

   
    pretrained_model = FxTTransformerLM(args)
    model_ckpt = os.path.join(args.pretrained_path, "model.safetensors")
    pretrained_model.load_state_dict(load_file(model_ckpt))


    tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_path,
                extra_ids=0,
                cache_dir=args.cache_dir,
                add_eos_token=False,
                additional_special_tokens=args.script_tokens
            )
    pretrained_model.tokenizer = tokenizer


    return pretrained_model, tokenizer


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
        "--work_dir", required=True, type=str, help="Directory for the results"
    )

    dataset = parser.add_argument_group("dataset setup")
    dataset.add_argument(
        "--dataset_name", type=str, help="Name of dataset on huggingface"
    )
    dataset.add_argument("--language", type=str, help="Language")
    dataset.add_argument(
        "--joint_input",
        type=bool,
        help="Whether to encode muliple inputs as a single sequence",
    )
    dataset.add_argument(
        "--cache_dir",
        type=str,
        default="/path/to/hf_cache",
        help="Directory to cache the dataset and tokenizer",
    )

    model = parser.add_argument_group("model setup")
    model.add_argument("--n_labels", type=int, default=3, help="Number of labels")
    model.add_argument(
        "--pretrained_path",
        type=str,
        help="Path to the pretrained model",
        required=True,
    )
    model.add_argument("--model_type", type=str, help="If model is fixed or routed")
    model.add_argument("--scale_bp", type=int, default=10, help="Scaling factor for boundary prediction loss")

    opt = parser.add_argument_group("optimizer setup")
    opt.add_argument(
        "--optim", default="adam", type=str, choices=["adam"], help="Optimizer to use"
    )
    opt.add_argument("--lr", type=float, help="Initial learning rate")
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
        "--max_train_steps", type=int, default=None, help="Max number of training steps"
    )
    training.add_argument(
        "--batch_size", type=int, default=16, help="Global batch size"
    )
    training.add_argument("--seed", type=int, default=42, help="Random seed")
    training.add_argument(
        "--seq_len", type=int, default=512, help="Maximum sequence length"
    )
    training.add_argument("--report_to", type=str, default="wandb", help="Wandb")
    training.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient_accumulation_steps",
    )
    training.add_argument(
        "--num_warmup_steps", type=int, default=5000, help="num_warmup_steps"
    )
    training.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup_ratio")
    training.add_argument(
        "--logging_steps", type=int, default=500, help="logging_steps"
    )
    training.add_argument(
        "--checkpointing_steps",
        type=str,
        help="whether to group texts ?",
        default="4000",
    )
    training.add_argument(
        "--with_tracking", type=bool, help="whether to track with wandb ?", default=False
    )
    training.add_argument(
        "--streaming", type=bool, help="whether to use streaming dataset", default=False
    )
    training.add_argument(
        "--resume_from_checkpoint",
        help="resume_from_checkpoint",
        default=None,
    )
    training.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    training.add_argument(
        "--scale_loss2",
        type=float,
        default=0.8,
        help="Scaling factor for the second loss term (if applicable)",
    )
    training.add_argument(
        "--use_best_model",
        type=str,
        default="false",
        help="Whether to use the best model based on validation loss for test evaluation. If False, uses the final model.",
    )
    training.add_argument(
        "--freeze_bp",
        type=str,
        default="false",
        help="Whether to freeze the script to boundary predictor during fine-tuning.",
    )

    parser.set_defaults(**config)

    args, _ = parser.parse_known_args()
    args.use_best_model = True if args.use_best_model.lower() == "true" else False

    return args


def main():
    args = parse_args()

    args.freeze_bp = True if args.freeze_bp.lower() == "true" else False

    set_seed(args.seed)
    config_file = os.path.join(args.pretrained_path, "config.json")
    config = read_json_file(config_file)
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.num_proc = 20
    # Create output directory with timestamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = "_".join(args.pretrained_path.split("/")[-3:])
    work_dir = f"{args.work_dir}/{str(model_dir).split('/')[-1]}/{str(args.dataset_name).replace('/', '_')}/epochs_{str(args.num_train_epochs)}/{args.language}/bz{args.batch_size}_seed{args.seed}"

    basename = f"{os.path.basename(work_dir)}_{current_time}"
    new_path = os.path.join(os.path.dirname(work_dir), basename)
    args.output_dir = new_path
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 50)
    print("Start training to {}".format(args.output_dir))

    # Create directory for best model checkpoint
    best_model_dir = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    # Accelerate config
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["project_dir"] = args.output_dir
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the sssconfiguration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    transformers.logging.set_verbosity_error()

    # Load pretrained model
    logger.info("Loading pretrained model ....")

    model, tokenizer = load_pretrained_model(args)
    if not args.resume_from_checkpoint:
        torch._dynamo.config.optimize_ddp = False
        model = torch.compile(model, dynamic=True)
    blocks = eval(args.model_config)
    aux_decoder = blocks[3]
    args.aux_num_epochs = 0

    tokenizer.save_pretrained(args.output_dir)

    script_to_id = {script: id_ for id_, script in args.id_to_script.items()}
    args.language_to_script_id = {
        lang: int(script_to_id[script]) for lang, script in args.language_to_script.items()
    }
    logger.info(f"language_to_script_id is {args.language_to_script_id}")

    ###########################################################################
    # Load data
    ###########################################################################
    logger.info("Loading data corpus ....")
    data_corpus = JointInputcorpus(
        language=args.language,
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        max_seq_length=args.seq_len,
        accelerator=accelerator,
        cache_dir=args.cache_dir,
        language_to_script_id=args.language_to_script_id,
        args=args,
    )

    
    # Save config file
    save_args_to_json(args, args.output_dir)

    data_collator = lambda x: dynamic_padding_data_collator(
        x, tokenizer
    )  
    # data_collator = default_data_collator
    
    train_dataloader = StatefulDataLoader(
        data_corpus.train_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        pin_memory=True,
        prefetch_factor=4,
        num_workers=args.num_proc,
        persistent_workers=args.num_proc > 0,
        shuffle=True,
    )

    eval_dataloader = StatefulDataLoader(
        data_corpus.validation_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        pin_memory=True,
        prefetch_factor=4,
        num_workers=args.num_proc,
        persistent_workers=args.num_proc > 0,
    )
    test_dataloader = StatefulDataLoader(
        data_corpus.test_dataset,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        pin_memory=True,
        prefetch_factor=4,
        num_workers=args.num_proc,
        persistent_workers=args.num_proc > 0,
    )
    # Initialize Classification model
    logger.info("Initializing model ....")
    logger.info(model)
    if args.freeze_bp:
        bp = model.script_to_bp_layers
        for p in bp.parameters():
            p.requires_grad = False
            
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_b1, args.adam_b2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    # train_dataloader is NOT passed to accelerator.prepare() — it's a StatefulDataLoader
    # whose state we save/restore for fast checkpoint resumption.
    model, optimizer, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, eval_dataloader, test_dataloader, scheduler
    )
    # freeze script_to_bp_layers
    

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)


    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "scheduler"
        ]  # .value
        accelerator.init_trackers(
            project_name="fxt-mtp",
            config=experiment_config,
            init_kwargs={"wandb": {"entity": os.environ.get("WANDB_ENTITY"), "name": basename}},
        )
        # pass
    
    # Train!
    n_params = count_trainable_parameters(model)
    logger.info(
        f"Training new model from scratch - Total size={n_params/1000000:.2f}M params"
    )
    total_batch_size = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(data_corpus.train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )

    completed_steps = 0
    starting_epoch = 0
    if args.resume_from_checkpoint is not None:
        checkpoint_path = args.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        torch._dynamo.config.optimize_ddp = False
        model = torch.compile(model, dynamic=True)
        

        # Restore StatefulDataLoader position — O(1), no sample iteration needed
        dl_state_path = os.path.join(
            checkpoint_path, f"dataloader_state_{accelerator.process_index}.pt"
        )
        if os.path.exists(dl_state_path):
            train_dataloader.load_state_dict(torch.load(dl_state_path, weights_only=True))
            logger.info(f"Restored dataloader state from {dl_state_path}")
        else:
            logger.warning(f"No dataloader state found at {dl_state_path}, starting from beginning of epoch")

        training_difference = os.path.splitext(os.path.basename(checkpoint_path))[0]
        if "epoch" in training_difference:
            # format: epoch_{N} or epoch_{N}_step_{M}
            parts = training_difference.split("_")
            starting_epoch = int(parts[1]) + 1
            completed_steps = int(parts[3]) if "step" in training_difference else starting_epoch * num_update_steps_per_epoch
        else:
            completed_steps = int(training_difference.replace("step_", ""))
            starting_epoch = completed_steps // num_update_steps_per_epoch

        logger.info(f"Resuming from epoch: {starting_epoch}, step: {completed_steps}")
    else:
        logger.info("No checkpoint found. Starting from scratch.")

    script_id = args.language_to_script_id[f"<{args.language}>"]
    logger.info("Evaluating model on test set before training")
    initial_test_metrics = (
        evaluate_model(
            model,
            test_dataloader,
            accelerator,
            args.batch_size,
            phase="test",
            task="SFT_eval",
            script_id=script_id,
        )
        
    )
    logger.info(f"Initial test metrics: {initial_test_metrics}")

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0  # Initialize total_loss for all cases
        # StatefulDataLoader resumes from its restored state automatically — no skipping needed.
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            # Pick first example, find prompt_len boundary
            with accelerator.accumulate(model):
                seq_loss, seq_loss2, stats, aux_loss, _ = model(batch, task="SFT", script_id=script_id)
                boundary_loss = aux_loss[0]

                if aux_decoder > 0:
                    loss = (seq_loss * 1) + args.scale_bp * boundary_loss + (args.scale_loss2 * seq_loss2)
                else:
                    loss = seq_loss + args.scale_bp * boundary_loss

                # We keep track of the loss at each epoch
                total_loss += loss.detach().float()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if step % 50 == 0:
                        grad_norm_ = grad_norm(model)
                        block_grad_norms_ = block_grad_norms(model)
                    current_lr = lr_scheduler.get_last_lr()[0]
                    accelerator.clip_grad_norm_(model.parameters(), args.clip)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if step % 50 == 0:
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "train_seq_loss": seq_loss,
                                "train_seq_loss2": seq_loss2,
                                "train_boundary_loss": boundary_loss,
                                "learning_rate": current_lr,
                            },
                            step=completed_steps,
                        )
                        accelerator.log(
                            stats,
                            step=completed_steps,
                        )
                        accelerator.log(
                            {
                            "grad_norm": grad_norm_,
                            **block_grad_norms_
                            }
                        )


                    logger.info(f"stats are {stats}")

            # if isinstance(checkpointing_steps, int):
            #     if completed_steps % checkpointing_steps == 0:
            #         output_dir = f"step_{completed_steps}"
            #         if args.output_dir is not None:
            #             output_dir = os.path.join(args.output_dir, output_dir)
            
            if completed_steps % int(args.checkpointing_steps * 2) == 0 and step != 0:
                output_dir = f"epoch_{epoch}_step_{completed_steps}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    save_clean_model_weights(accelerator, model, output_dir)
                    torch.save(
                        train_dataloader.state_dict(),
                        os.path.join(output_dir, f"dataloader_state_{accelerator.process_index}.pt"),
                    )
                    logger.info(f"Saved state to {output_dir}")
            
            if completed_steps >= args.max_train_steps:
                break
        ##########################################
        # Evaluate on validation set
        ##########################################
        logger.info(f"Evaluating validation set for epoch {epoch}")
        if args.output_dir is not None:
            output_dir = f"epoch_{epoch}"
            output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            save_clean_model_weights(accelerator, model, output_dir)
            torch.save(
                train_dataloader.state_dict(),
                os.path.join(output_dir, f"dataloader_state_{accelerator.process_index}.pt"),
            )
        eval_metrics = evaluate_model(
            model, eval_dataloader, accelerator,  args.batch_size, phase="eval", task="SFT_eval",
            script_id=script_id,
        )
        eval_loss = eval_metrics["eval_lm_loss"]
        eval_loss2 = eval_metrics["eval_lm_loss2"]
        eval_aux_loss = eval_metrics["eval_aux_loss"]

        logger.info(f"epoch {epoch}: eval_aux_loss {eval_aux_loss} valid loss {eval_loss} valid loss2 {eval_loss2}")

        metrics_dict = {
            "train_loss": total_loss.item() / len(train_dataloader),
            "eval_loss": eval_loss,
            "eval_loss2": eval_loss2,
            "epoch": epoch,
            "step": completed_steps,
        }

        if args.with_tracking:
            accelerator.log(
                metrics_dict,
                step=completed_steps,
            )

        output_dir = f"epoch_{epoch}"
        

    ##########################################
    # Aux-decoder fine-tuning (1 extra epoch)
    # Freeze entire model; only train blocks[3]
    ##########################################
    if args.aux_num_epochs > 0 and aux_decoder > 0:
        logger.info(f"aux_decoder > 0: freezing model and fine-tuning aux_decoder for {args.aux_num_epochs} epochs")

        # Freeze everything
        unwrapped = accelerator.unwrap_model(model)
        for param in unwrapped.parameters():
            param.requires_grad = False

        # Unfreeze only the aux-decoder stack (blocks[3])
        for param in unwrapped.blocks[3].parameters():
            param.requires_grad = True

        trainable = [p for p in unwrapped.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters for aux-decoder phase: {sum(p.numel() for p in trainable):,}")

        aux_optimizer = optim.Adam(
            trainable,
            lr=args.lr,
            betas=(args.adam_b1, args.adam_b2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        aux_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        aux_steps = aux_steps_per_epoch * args.aux_num_epochs
        aux_scheduler = get_scheduler(
            name="cosine",
            optimizer=aux_optimizer,
            num_warmup_steps=int(aux_steps * args.warmup_ratio),
            num_training_steps=aux_steps,
        )
        aux_optimizer, aux_scheduler = accelerator.prepare(aux_optimizer, aux_scheduler)

        model.train()
        aux_total_loss = 0
        aux_completed = 0
        aux_progress = tqdm(
            range(aux_steps),
            desc="aux-decoder epochs",
            disable=not accelerator.is_local_main_process,
        )
        for aux_epoch in range(args.aux_num_epochs):
            for step, batch in enumerate(train_dataloader):
                seq_loss, seq_loss2, stats, aux_loss, _ = model(batch, task="SFT")
                boundary_loss = aux_loss[0]

                loss = seq_loss2

                aux_total_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)

                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    accelerator.clip_grad_norm_(trainable, args.clip)
                    aux_optimizer.step()
                    aux_scheduler.step()
                    aux_optimizer.zero_grad()
                    aux_progress.update(1)
                    aux_completed += 1


                if step % args.logging_steps == 0:
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "aux_train_loss": seq_loss2,
                                "aux_boundary_loss": boundary_loss,
                                "learning_rate": aux_scheduler.get_last_lr()[0],
                            },
                            step=completed_steps,
                        )
                        accelerator.log(
                            stats,
                            step=completed_steps,
                        )
                        logger.info(f"stats are {stats}")
                if aux_completed >= aux_steps:
                    break

            # Evaluate and save after each aux epoch
            aux_epoch_eval_metrics = evaluate_model(
                model, eval_dataloader, accelerator, args.batch_size, phase="eval", task="SFT_eval",
                script_id=script_id,
            )
            logger.info(
                f"aux-decoder epoch {aux_epoch}: eval_loss {aux_epoch_eval_metrics['eval_lm_loss']} "
                f"eval_loss2 {aux_epoch_eval_metrics['eval_lm_loss2']}"
            )
            aux_epoch_dir = os.path.join(best_model_dir, f"aux_decoder_epoch_{aux_epoch}")
            os.makedirs(aux_epoch_dir, exist_ok=True)
            save_clean_model_weights(accelerator, model, aux_epoch_dir)
            logger.info(f"Saved aux-decoder epoch {aux_epoch} checkpoint to {aux_epoch_dir}")
            if args.with_tracking:
                accelerator.log(
                    {
                        "aux_epoch_eval_loss": aux_epoch_eval_metrics["eval_lm_loss"],
                        "aux_epoch_eval_loss2": aux_epoch_eval_metrics["eval_lm_loss2"],
                    },
                    step=completed_steps,
                )

            # Evaluate after aux-decoder epochs
            aux_eval_metrics = evaluate_model(
                model, eval_dataloader, accelerator, args.batch_size, phase="eval", task="SFT_eval",
                script_id=script_id
            )
            logger.info(f"aux-decoder epochs: eval_loss {aux_eval_metrics['eval_lm_loss']} eval_loss2 {aux_eval_metrics['eval_lm_loss2']}")

            # Save aux-decoder checkpoint
            aux_decoder_dir = os.path.join(best_model_dir, "aux_decoder")
            os.makedirs(aux_decoder_dir, exist_ok=True)
            save_clean_model_weights(accelerator, model, aux_decoder_dir)
            logger.info(f"Saved aux-decoder checkpoint to {aux_decoder_dir}")

            if args.with_tracking:
                accelerator.log(
                    {
                        "aux_train_loss": aux_total_loss.item() / len(train_dataloader),
                        "aux_eval_loss": aux_eval_metrics["eval_lm_loss"],
                        "aux_eval_loss2": aux_eval_metrics["eval_lm_loss2"],
                    },
                    step=completed_steps,
                )

            # Restore all parameters to trainable for test evaluation
            for param in unwrapped.parameters():
                param.requires_grad = True


    accelerator.wait_for_everyone()

    
    ##########################################
    # Evaluate on the test set
    ##########################################
    
    test_metrics = evaluate_model(
        model,
        test_dataloader,
        accelerator,
        args.batch_size,
        phase="test",
        task="SFT_eval",
        script_id=script_id,
    )
    final_test_loss = test_metrics["test_lm_loss"]
    final_test_loss2 = test_metrics["test_lm_loss2"]
    test_aux_loss = test_metrics["test_aux_loss"]

    logger.info(f"epoch {epoch}: test_aux_loss {test_aux_loss} test loss {final_test_loss} test loss2 {final_test_loss2}")

    test_metrics_dict = {
        "test_aux_loss": test_aux_loss,
        "test_loss": final_test_loss,
        "test_loss2": final_test_loss2,
        "epoch": epoch,
        "step": completed_steps,
    }

    if args.with_tracking:
        accelerator.log(
            test_metrics_dict,
            step=completed_steps,
        )

    

    final_metrics_dict = {
        "test_aux_loss": test_aux_loss,
        "valid_aux_loss": eval_aux_loss, #BP
        "train_loss": total_loss.item() / len(train_dataloader),
        "valid_loss": eval_loss,
        "valid_loss2": eval_loss2,
        "test_loss": final_test_loss,
        "test_loss2": final_test_loss2,
    }

   
    # Save final comparison results showing before/after training metrics for test set only
    comparison_file = os.path.join(args.output_dir, "training_comparison.txt")
    with open(comparison_file, "w") as writer:
        writer.write(
            "***** Model Performance Comparison Before and After Training *****\n\n"
        )

        # Write all test metrics only
        writer.write("Test Metrics:\n")
        for key, value in initial_test_metrics.items():
            if isinstance(value, (int, float)):
                writer.write(f"  Pre-training {key}: {value}\n")

        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                writer.write(f"  Post-training {key}: {value}\n")

        # Calculate improvements for all numeric metrics
        writer.write("\nImprovement Summary:\n")
        for key in initial_test_metrics.keys():
            if (
                key in test_metrics
                and isinstance(initial_test_metrics[key], (int, float))
                and isinstance(test_metrics[key], (int, float))
            ):
                improvement = test_metrics[key] - initial_test_metrics[key]
                writer.write(f"  {key} improvement: {improvement:.4f}\n")

    logger.info(f"Training comparison results saved at {comparison_file}")

    # Include all initial metrics in the final metrics dictionary with pre_training_ prefix
    for key, value in initial_test_metrics.items():
        final_metrics_dict[f"pre_training_{key}"] = value

    # Calculate and include improvements for all metrics
    for key in initial_test_metrics.keys():
        if (
            key in test_metrics
            and isinstance(initial_test_metrics[key], (int, float))
            and isinstance(test_metrics[key], (int, float))
        ):
            improvement = test_metrics[key] - initial_test_metrics[key]
            final_metrics_dict[f"{key}_improvement"] = improvement

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        save_clean_model_weights(accelerator, model, args.output_dir)

        # save results into a json file
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(final_metrics_dict, f)

    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()
