from __future__ import annotations

from datasets   import Dataset, load_dataset
from pathlib    import Path

import pandas as pd


REQUIRED_COLUMNS: set[str] = {
    "sample_id",
    "document",
    "summary"
}


class DatabaseError(Exception):
    """Exception raised for when the database cannot be loaded or has an unexpected error."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def load_evaluation_manifest(path: str | Path) -> Dataset:
    """
    Load a previously created evaluation manifest (see `src/eval_manifest.py`).
    """

    path: Path = Path(path)
    if not path.exists():
        raise DatabaseError(f"Evaluation manifest not found: {path}")

    try:
        dataframe: pd.DataFrame = pd.read_csv(path)
    except Exception as e:
        raise DatabaseError(f"Could not read evaluation manifest: {path}") from e

    missing_cols: set[str] = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_cols:
        raise DatabaseError(f"Evaluation manifest is missing columns: {sorted(missing_cols)}")

    if dataframe.empty:
        raise DatabaseError(f"Evaluation manifest is empty: {path}")

    if dataframe["sample_id"].duplicated().any():
        raise DatabaseError(f"Duplicate sample IDs found in: {path}")

    if dataframe["document"].isna().any():
        raise DatabaseError(f"Missing documents in manifest: {path}")

    if dataframe["summary"].isna().any():
        raise DatabaseError(f"Missing references in manifest: {path}")

    dataframe = dataframe[
        [
            "sample_id",
            "document",
            "summary"
        ]
    ]

    return Dataset.from_pandas(dataframe, preserve_index = False)

def load_dataset_subset(
        database_name: str,
        *,
        database_split: str     = "test",
        config: str | None      = None,
        num_samples: int | None = None,
        seed: int               = 42
) -> Dataset:
    """
    Load a dataset split and select a random subset. (fixed seed)

    Args:
        database_name: Name of the dataset.
        database_split: Which database split to take.
        config: (can be NULL).
        num_samples: How many samples to take (can be NULL).
        seed: Random seed.
    
    Returns:
        Random subset.
    """
    try:
        if config is not None:
            dataset = load_dataset(
                database_name,
                config,
                split = database_split
            )
        else:
            dataset = load_dataset(
                database_name,
                split = database_split
            )
    except Exception as e:
        raise DatabaseError(
            f"Could not download dataset '{database_name}' "
            f"(split = '{database_split}', config = '{config})"
        ) from e

    if num_samples is not None:
        if num_samples > len(dataset):
            raise ValueError(f"Requested {num_samples} samples, but only contains {len(dataset)}")

        dataset = dataset.shuffle(seed=seed).select(range(num_samples))

    return dataset
        

