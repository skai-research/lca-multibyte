import numpy as np
import torch
from typing import Optional
from multiprocessing import cpu_count

from accelerate.logging import get_logger
from datasets import load_dataset, interleave_datasets
from datasets import DatasetDict, IterableDatasetDict
from transformers import AutoTokenizer


logger = get_logger(__name__)


BYTE_MODEL = "google/byt5-small"
CACHE_DIR = "cache"
SEED = 42

np.set_printoptions(suppress=True)

ISO_MAPPING = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "fr": "fra_Latn",
}
TWO_LETTER_LANGUAGES = {
    "en": "english",
    "es": "spanish",
    "ru": "russian",
    "uk": "ukrainian",
    "hi": "hindi",
    "te": "telugu",
    "fr": "french",
}
TWO_LETTER_THREE_LETTER_LANGUAGES = {
    "en": "eng",
    "es": "spa",
    "ru": "rus",
    "uk": "ukr",
    "hi": "hin",
    "te": "tel",
    "fr": "fra",
}
ISO_DATASET = {
    "en": "HuggingFaceFW/fineweb",
    "es": "HuggingFaceFW/fineweb-2",
    "ru": "HuggingFaceFW/fineweb-2",
    "uk": "HuggingFaceFW/fineweb-2",
    "hi": "HuggingFaceFW/fineweb-2",
    "te": "HuggingFaceFW/fineweb-2",
}


def dynamic_padding_data_collator(features, tokenizer):
    """
    Dynamically pads sequences in the batch to the maximum length of the batch.
    """
    # Extract input_ids and attention_mask
    input_ids = [f["input_ids"] for f in features]
    attention_masks = [f["attention_mask"] for f in features]

    # Dynamically pad to the maximum sequence length in the batch
    batch = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,
        return_tensors="pt",
    )

    if "label" in features[0]:
        labels = [f["label"] for f in features]
        labels =  tokenizer.pad(
            {"input_ids": labels}, padding=True, return_tensors="pt"
        )
        batch['labels'] = labels["input_ids"]

    if "prompt_len" in features[0]:
        batch["prompt_len"] = torch.tensor(
            [f["prompt_len"] for f in features], dtype=torch.long
        )

    return batch

class MixtureByteVocab(object):
    """
    Create Byte Vocabulary
    """

    def __init__(self, **kwargs):
        tokenizer_path = kwargs.get("tokenizer_path", BYTE_MODEL)
        
        if tokenizer_path != BYTE_MODEL:
            kwargs["script_tokens"] = []

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            extra_ids=0,
            cache_dir=kwargs["cache_dir"],
            additional_special_tokens=kwargs["script_tokens"],
            use_fast=True,
        )
        print("Loaded tokenizer")
        self.script_to_id = kwargs["script_tokens"]
        
    @property
    def vocab_size(self):

        vocab_size = 0 #max(self.tokenizer.added_tokens_decoder.keys())
        if vocab_size == 0:
            vocab_size = len(self.tokenizer)
            # self.tokenizer.pad_token = self.tokenizer.eos_token

        else:
            vocab_size = vocab_size + 1
        return vocab_size

    def __len__(self):
        return self.vocab_size


class JointInputcorpus(object):
    def __init__(
        self,
        language,
        dataset_name,
        tokenizer,
        max_seq_length,
        accelerator,
        cache_dir,
        args,
        language_to_script_id: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.language_to_script_id = language_to_script_id
        args.num_proc = max(args.num_proc, cpu_count()- 4)


        if args.task_type == "SFT":
            train_dict, validation_dict, test_dict = (
                DatasetDict(),
                DatasetDict(),
                DatasetDict(),
            )

            for lang in language.split(","):
                if args.dataset_name == "tulu": #allenai/tulu-v2-sft-mixture
                    dataset = load_dataset("allenai/tulu-3-sft-mixture", cache_dir=cache_dir)
                    #filter out sources= [ai2-adapt-dev/oasst1_converted, ]
                    dataset = dataset.filter(lambda x: x["source"] not in ["ai2-adapt-dev/coconot_converted", "ai2-adapt-dev/oasst1_converted", "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k", "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k", ])
                    #shuffle and use the last 1k as validation and the rest as train
                    dataset = dataset.shuffle(seed=SEED)
                    dataset['validation'] = dataset['train'].select(range(1000))
                    dataset['train'] = dataset['train'].select(range(1000, len(dataset['train'])))
                elif "sum" in args.dataset_name or "cnn" in args.dataset_name:
                    #rename columns
                    if args.dataset_name == "cnn_dailymail":
                        dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir=cache_dir)
                        dataset = dataset.rename_columns({"article": "inputs", "highlights": "targets"})
                elif args.dataset_name == "opus-100":
                    dataset = load_dataset(
                    "Helsinki-NLP/opus-100", f"{args.target}-{args.source}", cache_dir=cache_dir, trust_remote_code=True
                )

                # Shard train across GPUs so each process only preprocesses its portion
                if accelerator.num_processes > 1:
                    dataset['train'] = dataset['train'].shard(
                        num_shards=accelerator.num_processes,
                        index=accelerator.process_index,
                    )
                dataset = dataset.shuffle(seed=SEED)
                
                with accelerator.main_process_first():
                    
                    if "sum" in args.dataset_name or "cnn" in args.dataset_name:
                        dataset = dataset.map(
                            self.preprocess_sum,
                            batched=True,
                            desc="Preprocessing summarization data",
                            load_from_cache_file=True,
                            num_proc=args.num_proc,
                            )
                    elif args.dataset_name == "tulu":
                        dataset = dataset.map(
                            self.preprocess_sft,
                            batched=True,
                            desc="Preprocessing SFT data",
                            load_from_cache_file=True,
                            num_proc=args.num_proc,
                            )

                    elif args.dataset_name == "opus-100":
                        print("Preprocessing translation data for language pair {}-{}".format(args.source, args.target))
                        dataset = dataset.map(
                            self.preprocess_translation,
                            fn_kwargs={"args": args},
                            batched=True,
                            desc=f"Running column splitting for {dataset_name} dataset",
                            load_from_cache_file=False,
                        )
                        

                    # dataset['train'] = dataset['train'].select(range(5000))
                    dataset['validation'] = dataset['validation'].select(range(min(1000, len(dataset['validation']))))
                    dataset['test'] = dataset['validation']


                    tokenized_datasets = dataset.map(
                        self.process_sft, batched=True,
                        remove_columns=dataset["train"].column_names,
                        desc="Running tokenizer on dataset",
                        load_from_cache_file=False,
                        num_proc=args.num_proc,
                    )

                    tokenized_datasets = tokenized_datasets.filter(self.filter_max_length, num_proc=args.num_proc)
                    if args.streaming:
                        tokenized_datasets = IterableDatasetDict(
                            {
                                split: tokenized_datasets[split].to_iterable_dataset()
                                for split in tokenized_datasets.keys()
                            }
                        )

                train_dict[lang] = tokenized_datasets["train"]
                validation_dict[lang] = tokenized_datasets["validation"]
                test_dict[lang] = tokenized_datasets["test"]

        if len(train_dict) < 2:
            self.train_dataset = train_dict[language]
            self.validation_dataset = validation_dict[language]
            self.test_dataset = test_dict[language]
        else:
            self.train_dataset = interleave_datasets(
                train_dict.values(), seed=SEED, stopping_strategy="first_exhausted"
            )
            self.validation_dataset = interleave_datasets(
                validation_dict.values(), seed=SEED, stopping_strategy="all_exhausted"
            )
            self.test_dataset = interleave_datasets(
                test_dict.values(), seed=SEED, stopping_strategy="all_exhausted"
            )
        self.individual_validation_dataset = validation_dict
        self.individual_test_dataset = test_dict


                
    def filter_max_length(self, example):
        return len(example["input_ids"]) <= self.max_seq_length and len(example["label"]) <= self.max_seq_length
    

    def preprocess_translation(self, examples, args):
        """
        Preprocess the translation dataset by formatting the text for translation.
        """

        examples['source'] = [example[args.source] for example in examples['translation']]
        examples['targets'] = [f"{example[args.target]}" for example in examples['translation']]

        if args.dataset_name == "flores":
            examples['inputs'] = [f"<{TWO_LETTER_LANGUAGES[args.source]}> {example} <{TWO_LETTER_LANGUAGES[args.target]}> " for example in examples['source']]
        else:
            examples['inputs'] = [f"<{TWO_LETTER_LANGUAGES[args.source]}> {example[args.source]} <translate to {TWO_LETTER_LANGUAGES[args.target]}> {example[args.target]}" for example in examples['translation']]


        return examples

    def process_translation(self, examples, args, split):


        #add translation template to source and target language
        #format

        if split == "train":
            examples['text'] = [f"<{TWO_LETTER_LANGUAGES[args.source]}> {example[args.source]} <translate to {TWO_LETTER_LANGUAGES[args.target]}> {example[args.target]}" for example in examples['translation']]
            tokenized_inputs = self.tokenizer(
                examples["text"],
                truncation=False,
                max_length=self.max_seq_length,
                add_special_tokens=True,
            )

        else:
            if args.dataset_name == "flores":
                examples['text'] = [f"<{TWO_LETTER_LANGUAGES[args.source]}> {example} <{TWO_LETTER_LANGUAGES[args.target]}> " for example in examples['source']]
            else:
                examples['text'] = [f"<{TWO_LETTER_LANGUAGES[args.source]}> {example[args.source]} <translate to {TWO_LETTER_LANGUAGES[args.target]}> " for example in examples['translation']]
            tokenized_inputs = self.tokenizer(
                examples["text"],
                truncation=False,
                max_length=self.max_seq_length,
                add_special_tokens=False,
            )


        labels = self.tokenizer(
            examples["target"],
            truncation=False,
            max_length=self.max_seq_length,
            add_special_tokens=True,
            padding_side="right",
        )
        tokenized_inputs["label"] = labels["input_ids"]

        tokenized_inputs["prompt_len"] = [
            len(input_ids) - len(label_ids)
            for input_ids, label_ids in zip(tokenized_inputs["input_ids"], labels["input_ids"])
        ]
        
        return tokenized_inputs

    def preprocess_sum(self, examples):
        # add space in from of target
        examples["targets"] = [
            " " + target if not target.startswith(" ") else target
            for target in examples["targets"]
        ]
        # change inputs to inputs + target for sft
        examples["inputs"] = [
            input_text  + " <summarize>" + target #remember to take this out for Aya SFT
            for input_text, target in zip(examples["inputs"], examples["targets"])
        ]
        return examples
    
    def preprocess_sft(self, examples):
        # add space in from of target
        inputs = []
        targets = []
        for  i in range(len(examples["messages"])):
            targets.append("\n" + examples['messages'][i][-1]['content'])
            inputs.append(examples['messages'][i][0]['content'] + targets[i])


        examples["inputs"] = inputs
        examples["targets"] = targets
        return examples

    
    

    def process_sft(self, examples):
        tokenized_data = self.tokenizer(
            examples["inputs"],
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        labels = self.tokenizer(
            examples["targets"],
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )

        tokenized_data["label"] = labels["input_ids"]

        tokenized_data["prompt_len"] = [
            len(input_ids) - len(label_ids)
            for input_ids, label_ids in zip(tokenized_data["input_ids"], labels["input_ids"])
        ]

        return tokenized_data
    


