from __future__ import annotations

import os
import warnings
import pandas as pd
import transformers

from dataclasses    import dataclass
from pathlib        import Path
from datasets       import Dataset

from src.db_loader      import load_dataset_subset
from src.model_method   import ModelPrecision

import src.fp16 as fp16
import src.gguf as gguf
import src.mlx  as mlx

# ! takes too much computation time
# import src.bnb as bnb


warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()


@dataclass(frozen=True)
class ExperimentalConfig:
    """
    Global benchmark configuration.
    """

    sample_size: int    = 100
    seed: int           = 42

    output_dir: Path    = Path("results")
    samples_dir: Path   = Path("data")

    max_new_tokens: int     = 128
    max_input_tokens: int   = 1024


CONFIG = ExperimentalConfig()


def load_datasets(config: ExperimentalConfig) -> tuple[Dataset, Dataset]:
    """
    Load the datasets used for the project. (CNN + XSUM).
    """

    cnn = load_dataset_subset(
        database_name   = "abisee/cnn_dailymail",
        config          = "3.0.0",
        database_split  = "test",
        num_samples     = config.sample_size,
        seed            = config.seed
    )

    xsum = load_dataset_subset(
        database_name   = "EdinburghNLP/xsum",
        database_split  = "test",
        num_samples     = config.sample_size,
        seed            = config.seed
    )

    cnn = cnn.rename_column("article", "document")
    cnn = cnn.rename_column("highlights", "summary")

    return cnn, xsum


def load_models(config: ExperimentalConfig) -> list[ModelPrecision]:
    """
    Make a list of the defined models for the expleriment.
    """

    models: list[ModelPrecision] = [
        fp16.ModelFP16(),
        mlx.ModelMLX(),
        gguf.ModelGGUF()
    ]

    return models


def save_examples(
        dataset: Dataset,
        models: list[ModelPrecision],
        dataset_name: str,
        config: ExperimentalConfig
):
    """
    Save documents and model predictions.
    """

    rows = {
        "Document":             dataset["document"],
        "Human-made Reference": dataset["summary"]
    }

    for model in models:
        if len(model.predictions) != len(dataset):
            raise RuntimeError(
                f"Predictions count mismatch for model '{model._model_name}': "
                f"{len(model.predictions)} predictions for {len(dataset)} dataset samples."
            )

        rows[model._model_name] = model.predictions

    examples: pd.DataFrame = pd.DataFrame(rows)

    config.samples_dir.mkdir(parents=True, exist_ok=True)

    output_path: Path = (
        config.samples_dir
        / f"qualitative_examples_{dataset_name}.csv"
    )

    examples.to_csv(output_path, index=False)

    print(f"✓ Samples saved in {output_path}")


def run_models(
        dataset: Dataset,
        dataset_name: str,
        config: ExperimentalConfig
):
    """
    Run every model on the same dataset.
    """

    print()
    print("=" * 70)
    print(f"DATASET: {dataset_name}")
    print(f"SAMPLES: {len(dataset)}")
    print("=" * 70)

    models: list[ModelPrecision] = load_models(config)

    all_results: list[dict] = []

    for model in models:
        print()
        print("-" * 70)
        print(f"MODEL: {model.model_name}")
        print("-" * 70)

        model.max_tokens = config.max_new_tokens

        result = model.run(
            dataset,
            dataset_name
        )

        result.update(
            {
                "Seed":             config.seed,
                "Max input tokens": config.max_input_tokens,
                "Max new tokens":   config.max_new_tokens
            }
        )

        all_results.append(result)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path: Path = (
        config.output_dir
        / f"benchmark_results_{dataset_name}.csv"
    )

    pd.DataFrame(all_results).to_csv(
        benchmark_path,
        index = False
    )

    print()
    print(f"✓ Benchmark results saved to: {benchmark_path}")

    save_examples(dataset, models, dataset_name, config)


def main() -> None:
    print("=" * 70)
    print("QUANTIZATION SUMMARIZATION BENCHMARK")
    print("=" * 70)
    print(f"Samples per dataset : {CONFIG.sample_size}")
    print(f"Random seed         : {CONFIG.seed}")
    print(f"Max new tokens      : {CONFIG.max_new_tokens}")
    print(f"NF4 enabled         : {CONFIG.run_nf4}")

    cnn_dataset, xsum_dataset = load_datasets(CONFIG)

    run_models(
        cnn_dataset,
        "cnn",
        CONFIG,
    )

    run_models(
        xsum_dataset,
        "xsum",
        CONFIG,
    )


if __name__ == "__main__":
    main()
