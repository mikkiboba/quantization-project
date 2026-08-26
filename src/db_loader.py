import sys
from datasets import load_dataset, Dataset
import os


TEXT_REPEAT: int = 50


class DatabaseNotFoundError(Exception):
    """Exception raised for when the database is not found."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def load(db_name: str, db_version: str | None = None, db_split: str = "test", num_samples: int | None = None, seed: int = 42) -> Dataset:
    """Return the dataset. Slice if `num_samples` is specified

    Args:
        db_name: Name of the dataset.
        db_version: If needed, the version of the dataset.
        db_split: What portion of dataset to take (test, eval, ...)
        num_samples: How many samples to take from the dataset.
        seed: Random seed.

    Returns:
        The dataset.
    
    Raises:
        DatabaseNotFoundError if the dabased cannot be loaded.
    """
    print("-"*TEXT_REPEAT)
    print(f"> Loading dataset \"{db_name}\"...")
    try:
        args = [db_name]
        if db_version:
            args.append(db_version)
        ds: Dataset = load_dataset(*args, split=db_split)

        if num_samples and num_samples < len(ds):
            ds = ds.shuffle(seed=seed).select(range(num_samples))

        print(f"\t✔ Successfully loaded dataset \"{db_name}\".")
        print(f"\t- Sliced {db_split} instances: {len(ds)}.")
        print(f"\t- Columns: {ds.column_names}.")
        print("-"*TEXT_REPEAT+'\n')
        return ds
    except Exception:
        raise DatabaseNotFoundError(f"Failed to load database \"{db_name}\".")


def save_local_benchmark(num_samples: int = 200):
    """Prepare and save benchmark datasets locally in /data"""
    os.makedirs("data", exist_ok=True)
    print(f"> Preparing local benchmark datasets ({num_samples} samples)")

    cnn_ds = load("abisee/cnn_dailymail", db_version="3.0.0", num_samples=num_samples)
    cnn_ds.save_to_disk("data/cnn_dailymail_sample")

    xsum_ds = load("EdinburghNLP/xsum", num_samples=num_samples)
    xsum_ds.save_to_disk("data/xsum_sample")

    print(f"✔ Successfully saved datasets in /data folder.")
