from __future__ import annotations

from pathlib    import Path
from datasets   import load_dataset

import pandas as pd


TEXT_LEN: int       = 70
SEED: int           = 42
SAMPLE_SIZE: int    = 100

OUTPUT_DIR = Path("data")


def create_manifest(
        dataset_name: str,
        *,
        config: str | None,
        split: str,
        output_name: str,
        sample_size: int    = SAMPLE_SIZE,
        seed: int           = SEED
):
    """
    Donwload a dataset split, shuffle it, select a fixed number of examples, and save the resulting evaluation set.
    """

    print()
    print("=" * TEXT_LEN)
    print(f"Creating evaluation manifest: {output_name}")
    print("=" * TEXT_LEN)

    print(f"Dataset      : {dataset_name}")
    print(f"Config       : {config}")
    print(f"Split        : {split}")
    print(f"Seed         : {SEED}")
    print(f"Sample size  : {SAMPLE_SIZE}")

    if config is None:
        dataset = load_dataset(dataset_name, split = split)
    else:
        dataset = load_dataset(dataset_name, config, split = split)

    if sample_size > len(dataset):
        raise ValueError(
            f"Requested {sample_size} samples but dataset contains only {len(dataset)} elements."
        )

    dataset = dataset.shuffle(seed = seed).select(range(sample_size))

    if dataset_name == "abisee/cnn_dailymail":
        dataset = dataset.rename_column("article", "document")
        dataset = dataset.rename_column("highlights", "summary")

    sample_ids = list(range(len(dataset)))

    documents   = dataset["document"]
    references  = dataset["summary"]

    manifest = pd.DataFrame(
        {
            "sample_id":    sample_ids,
            "document":     documents,
            "summary":      references,
        }
    )

    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
    output_path = OUTPUT_DIR / output_name

    manifest.to_csv(output_path, index = False)

    print(f"Saved to    : {output_path}")
    print(f"Rows        : {len(manifest)}")

    if manifest["sample_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate sample IDs detected."
        )

    if manifest["document"].isna().any():
        raise RuntimeError(
            "At least one document is missing."
        )

    if manifest["summary"].isna().any():
        raise RuntimeError(
            "At least one reference summary is missing."
        )

    print("Validation   : OK")


def main() -> None:

    create_manifest(
        "abisee/cnn_dailymail",
        config="3.0.0",
        split="test",
        output_name="cnn_eval_seed42_100.csv",
    )

    create_manifest(
        "EdinburghNLP/xsum",
        config=None,
        split="test",
        output_name="xsum_eval_seed42_100.csv",
    )

    print()
    print("=" * TEXT_LEN)
    print("All evaluation manifests created successfully.")
    print("=" * TEXT_LEN)


if __name__ == "__main__":
    main()
