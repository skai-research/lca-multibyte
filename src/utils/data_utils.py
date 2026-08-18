import os
import math
import tempfile
import itertools
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from itertools import chain
from typing import Optional, Dict
from multiprocessing import cpu_count
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset, interleave_datasets, load_from_disk
from datasets import DatasetDict, IterableDatasetDict, Dataset, IterableDataset, Features
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
}
TWO_LETTER_LANGUAGES = {
    "en": "english",
    "es": "spanish",
    "ru": "russian",
    "uk": "ukrainian",
    "hi": "hindi",
    "te": "telugu",
}
TWO_LETTER_THREE_LETTER_LANGUAGES = {
    "en": "eng",
    "es": "spa",
    "ru": "rus",
    "uk": "ukr",
    "hi": "hin",
    "te": "tel",
}
ISO_DATASET = {
    "en": "HuggingFaceFW/fineweb",
    "es": "HuggingFaceFW/fineweb-2",
    "ru": "HuggingFaceFW/fineweb-2",
    "uk": "HuggingFaceFW/fineweb-2",
    "hi": "HuggingFaceFW/fineweb-2",
    "te": "HuggingFaceFW/fineweb-2",
}


def convert_iterable_to_dataset(iterable_dataset):
    """Converts an iterable dataset to a regular dataset.

    Args:
      iterable_dataset: The iterable dataset to convert.

    Returns:
      A Dataset object containing the data from the iterable dataset.
    """
    data_list = list(iterable_dataset)
    dataset = Dataset.from_list(data_list, features=iterable_dataset.features)
    return dataset


def batch_iterator(iterable, batch_size):
    """Yields batches of data from an iterable."""
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def write_batch_to_parquet(batch_data, features, file_path):
    """Converts a batch of data (list of dicts) to Arrow and writes to Parquet."""
    try:

        arrow_table = pa.Table.from_pylist(batch_data, schema=features.arrow_schema)
        pq.write_table(arrow_table, file_path)
        # print(f"Successfully wrote {len(batch_data)} records to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error writing batch to {file_path}: {e}")
        return None


def convert_iterable_to_dataset_multiprocess(
    iterable_dataset: IterableDataset,
    features: Features = None,
    batch_size: int = 20000,
    num_proc: int = None,
):
    """
    Converts an IterableDataset to a regular Dataset using multiprocessing
    by writing batches to temporary Parquet files and then loading them.

    Args:
      iterable_dataset: The iterable dataset to convert.
      batch_size: How many items to process in each batch per worker.
      num_proc: Number of worker processes. Defaults to cpu_count().

    Returns:
      A Dataset object containing the data.
    """
    if features is None:
        features = iterable_dataset.features
    if num_proc is None:
        num_proc = cpu_count() - 2

    processed_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        results = []

        for i, batch in tqdm(
            enumerate(batch_iterator(iterable_dataset, batch_size)),
            total=math.ceil(iterable_dataset.length / batch_size),
            desc="Processing batches",
        ):
            if not batch:
                continue
            file_path = os.path.join(temp_dir, f"batch_{i}.parquet")
            result = write_batch_to_parquet(batch, features, file_path)
            results.append(result)

        for file_path in results:
            if file_path and os.path.exists(file_path):
                processed_files.append(file_path)
            else:
                print("Warning: A batch failed to write or returned None.")

        if not processed_files:
            print("Error: No data was successfully processed.")
            return None

        print(
            f"Finished writing batches. Loading dataset from {len(processed_files)} files..."
        )
        # Load the dataset from all the generated Parquet files
        # Ensure correct features are passed if not inferred correctly
        final_dataset = Dataset.from_parquet(processed_files, features=features)

    print("Dataset conversion complete.")
    return final_dataset


def insert_special_token(example, script_id):
    """
    Insert script-id at the front of every sequence
    """
    example["input_ids"].insert(0, script_id)
    return example


def group_texts(examples, max_seq_length):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length

    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }

    return result


def load_dataset_from_disk(path, fs=None):
    """
    Load a dataset from disk using the provided filesystem.
    """
    if fs is not None:
        return fs.load_from_disk(path)
    else:
        return Dataset.load_from_disk(path)


def save_to_disk(dataset, save_dir, split):
    """
    Save a dataset to disk using the provided filesystem.
    """
    if isinstance(dataset, IterableDataset):
        dataset = convert_iterable_to_dataset_multiprocess(dataset, dataset.features)

    savepath = os.path.join(save_dir, split)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    dataset.save_to_disk(savepath)
    # load it back and convert to iterable dataset
    dataset = load_from_disk(savepath)
    dataset = dataset.to_iterable_dataset()

    return dataset


def transform_labels_to_ids(labels, label_list):
    """
    this take a huggingface dataset column with text label and transform it to a list of ids
    """
    label_to_id = {label: i for i, label in enumerate(label_list)}
    transformed_labels = [label_to_id[label] for label in labels]
    return transformed_labels


class MixtureByteVocab(object):
    """
    Create Byte Vocabulary
    """

    def __init__(self, **kwargs):
        tokenizer_path = kwargs.get("tokenizer_path", BYTE_MODEL)
        
        # if tokenizer_path != BYTE_MODEL:
        #     kwargs["script_tokens"] = []


        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            extra_ids=0,
            cache_dir=kwargs["cache_dir"],
            additional_special_tokens=kwargs["script_tokens"] if tokenizer_path == BYTE_MODEL else [],
            use_fast=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
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


class FxTDataset(object):
    def __init__(
        self,
        file_paths: str,
        seq_len: int,
        accelerator: Accelerator,
        language_to_script_id: Optional[Dict] = None,
        args=None,
        **kwargs,
    ):

        self.seq_len = seq_len
        self.vocab = MixtureByteVocab(**kwargs)
        self.language_to_script_id = language_to_script_id
        train_dict, validation_dict, test_dict = (
            DatasetDict(),
            DatasetDict(),
            DatasetDict(),
        )

        for file_path in language_to_script_id:
            # args.load_from_disk = False
            if args.load_from_disk:
                file_path = file_path.replace("<", "").replace(">", "")

                num_shards = max(1, args.num_proc)
                train = load_dataset("parquet", data_files=f"{os.path.join(args.data, file_path)}/*.parquet")['train'].to_iterable_dataset(num_shards=num_shards)
                
                validation = load_dataset("parquet", data_files=f"{os.path.join(args.data, 'validation')}/*.parquet")['train'].select(range(args.val_size[file_path])).to_iterable_dataset(num_shards=num_shards)
                test = load_dataset("parquet", data_files=f"{os.path.join(args.data, 'test')}/*.parquet")['train'].select(range(args.val_size[file_path])).to_iterable_dataset(num_shards=num_shards)
                
                dataset = IterableDatasetDict(
                    {"train": train, "validation": validation, "test": test}
                )

            else:

                if file_path == "en":
                    raw_dataset = load_dataset(
                        ISO_DATASET[file_path],
                        split="train",
                        cache_dir=args.cache_dir,
                        streaming=args.streaming,
                    )
                    dataset = raw_dataset.take(
                        args.train_size[file_path]
                        + args.val_size[file_path]
                        + args.val_size[file_path]
                    )
                    columns_to_remove = [
                        "id",
                        "dump",
                        "url",
                        "date",
                        "file_path",
                        "language",
                        "language_score",
                        "token_count",
                    ]
                    dataset = dataset.remove_columns(columns_to_remove)

                    validation = dataset.take(args.val_size[file_path])
                    train_test = dataset.skip(args.val_size[file_path])
                    train = train_test.take(args.train_size[file_path])
                    test = train_test.skip(args.train_size[file_path])
                    train.length = args.train_size[file_path]
                    validation.length = args.val_size[file_path]
                    test.length = args.val_size[file_path]

                    print(
                        "💰 Saving train and validation and test to disk, this might take a while"
                    )
                    # convert train and validation to dataset and save both to disk
                    save_dir = os.path.join(args.data, file_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    # Save the dataset to disk
                    test = save_to_disk(test, save_dir, "test")
                    train = save_to_disk(train, save_dir, "train")
                    validation = save_to_disk(validation, save_dir, "validation")

                    dataset = IterableDatasetDict(
                        {"train": train, "validation": validation, "test": test}
                    )

                else:
                    dataset = load_dataset(
                        ISO_DATASET[file_path],
                        name=ISO_MAPPING[file_path],
                        trust_remote_code=True,
                        cache_dir=args.cache_dir,
                        streaming=args.streaming,
                    )
                    columns_to_remove = [
                        "id",
                        "dump",
                        "url",
                        "date",
                        "file_path",
                        "language",
                        "language_score",
                        "language_script",
                        "minhash_cluster_size",
                        "top_langs",
                    ]
                    dataset = dataset.remove_columns(columns_to_remove)
                    validation = dataset["test"].take(args.val_size[file_path])
                    validation.length = args.val_size[file_path]
                    train = dataset["train"].take(args.train_size[file_path])
                    train.length = args.train_size[file_path]

                    test = dataset["test"].skip(args.val_size[file_path])
                    test = test.take(args.val_size[file_path])
                    test.length = args.val_size[file_path]

                    # convert train and validation to dataset and save both to disk
                    save_dir = os.path.join(args.data, file_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # Save the dataset to disk
                    # train = save_to_disk(train, save_dir, "train")
                    # validation = save_to_disk(validation, save_dir, "validation")
                    test = save_to_disk(test, save_dir, "test")

                    dataset = IterableDatasetDict(
                        {"train": train, "validation": validation, "test": test}
                    )

                if not args.streaming:
                    dataset = DatasetDict(
                        {
                            "train": dataset[0],
                            "validation": dataset[1],
                            "test": dataset[2],
                        }
                    )
            with accelerator.main_process_first():
                tokenized_datasets = dataset.map(
                    self.tokenize_group_function, batched=True, remove_columns=["text"]
                )

                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    fn_kwargs={"max_seq_length": self.seq_len},
                )

            train_dict[f"<{file_path}>"] = tokenized_datasets["train"]
            validation_dict[f"<{file_path}>"] = tokenized_datasets["validation"]
            test_dict[f"<{file_path}>"] = tokenized_datasets["test"]
        # concatenate all datasets and stream. Data from all languages willn stop streaming as soon as data from any language is exhausted.
        # If you want to keep streaming, change the stopping strategy to "last_exhausted"
        self.train_dataset = interleave_datasets(
            train_dict.values(), seed=SEED, stopping_strategy="all_exhausted"
        )
        self.validation_dataset = interleave_datasets(
            validation_dict.values(), seed=SEED, stopping_strategy="all_exhausted"
        )
        self.test_dataset = interleave_datasets(
            test_dict.values(), seed=SEED, stopping_strategy="all_exhausted"
        )
        self.individual_validation_dataset = validation_dict
        self.individual_test_dataset = test_dict

    def tokenize_group_function(self, examples):
        return self.vocab.tokenizer(examples["text"], return_special_tokens_mask=True, add_special_tokens=True)


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = 0 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(0)
        else:
            # Same word as previous token
            label = labels[word_id]
            # # If the label is B-XXX we change it to I-XXX
            # if label % 2 == 1:
            #     label += 1
            new_labels.append(label)

    return new_labels


