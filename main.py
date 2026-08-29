from __future__ import annotations

import os
import warnings
import pandas as pd
import transformers
import argparse

from dataclasses    import dataclass
from pathlib        import Path
from datasets       import Dataset

from src.db_loader      import load_dataset_subset, load_evaluation_manifest
from src.model_method   import ModelPrecision

import src.model_fp16 as fp16
import src.gguf as gguf
import src.model_mlx  as mlx

# ! takes too much computation time
# import src.bnb as bnb


TEXT_LEN: int = 70


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

    cnn_manifest: Path      = Path("data/cnn_eval_seed42_100.csv")
    xsum_manifest: Path = Path("data/xsum_eval_seed42_100.csv")


CONFIG = ExperimentalConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Run one quantization configuration."
    )
    parser.add_argument(
        "--model",
        required = True,
        choices = [
            "fp16",
            "mlx",
            "gguf",
            "gptq",
        ],
        help = "Model configuration to benchmark."
    )

    return parser.parse_args()


def load_datasets(config: ExperimentalConfig) -> tuple[Dataset, Dataset]:
    """
    Load the manifest.
    """

    cnn: Dataset    = load_evaluation_manifest(config.cnn_manifest)
    xsum: Dataset   = load_evaluation_manifest(config.xsum_manifest)

    return cnn, xsum


def _load_datasets(config: ExperimentalConfig) -> tuple[Dataset, Dataset]:
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


def load_model(model_name: str) -> ModelPrecision:
    """
    Create one model configuration.
    """

    if model_name == "fp16":
        return fp16.ModelFP16()
    if model_name == "mlx":
        return mlx.ModelMLX()
    if model_name == "gguf":
        raise NotImplementedError("GGUF has not been implemented yet.")
    if model_name == "gptq":
        raise NotImplementedError("GPTQ has not been implemented yet.")

    raise ValueError(f"Unknown model: {model_name}")


def load_models(config: ExperimentalConfig) -> list[ModelPrecision]:
    """
    Make a list of the defined models for the expleriment.
    """

    models: list[ModelPrecision] = []
    models.append(fp16.ModelFP16())
    models.append(mlx.ModelMLX())

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
                f"Predictions count mismatch for model '{model.model_name}': "
                f"{len(model.predictions)} predictions for {len(dataset)} dataset samples."
            )

        rows[model.model_name] = model.predictions

    examples: pd.DataFrame = pd.DataFrame(rows)

    config.samples_dir.mkdir(parents=True, exist_ok=True)

    output_path: Path = (
        config.samples_dir
        / f"qualitative_examples_{dataset_name}.csv"
    )

    examples.to_csv(output_path, index=False)

    print(f"✓ Samples saved in {output_path}")


def run_model(
        dataset: Dataset,
        dataset_name: str,
        model_name: str,
        config: ExperimentalConfig
):
    """
    Run a model on the same dataset.
    """

    print()
    print("=" * TEXT_LEN)
    print(f"DATASET: {dataset_name}")
    print(f"MODEL: {model_name}")
    print(f"SAMPLES: {len(dataset)}")
    print("=" * TEXT_LEN)

    model: ModelPrecision = load_model(model_name)

    model.max_tokens        = config.max_new_tokens
    model.max_inp_tokens    = config.max_input_tokens

    result = model.run(
        dataset,
        dataset_name
    )

    result.update(
        {
            "Seed":             config.seed,
            "Max input tokens": config.max_input_tokens,
            "Max new tokens":   config.max_new_tokens,
            "Evaluation manifest": (
                str(config.cnn_manifest) if dataset_name == "cnn"
                else str(config.xsum_manifest)
            )
        }
    )

    config.output_dir.mkdir(parents = True, exist_ok = True)

    benchmark_path: Path = config.output_dir / f"benchmark_{model_name}_{dataset_name}.csv"

    pd.DataFrame([result]).to_csv(benchmark_path, index = False)

    print()
    print(f"✓ Results saved to: {benchmark_path}")

    save_examples(dataset, [model], dataset_name, config)


def run_models(
        dataset: Dataset,
        dataset_name: str,
        config: ExperimentalConfig
):
    """
    Run every model on the same dataset.
    """

    print()
    print("=" * TEXT_LEN)
    print(f"DATASET: {dataset_name}")
    print(f"SAMPLES: {len(dataset)}")
    print("=" * TEXT_LEN)

    models: list[ModelPrecision] = load_models(config)

    all_results: list[dict] = []

    for model in models:
        print()
        print("-" * TEXT_LEN)
        print(f"MODEL: {model.model_name}")
        print("-" * TEXT_LEN)

        model.max_tokens        = config.max_new_tokens
        model.max_inp_tokens    = config.max_input_tokens

        result = model.run(
            dataset,
            dataset_name
        )

        result.update(
            {
                "Seed":                 config.seed,
                "Evaluation manifest": (
                                        str(config.cnn_manifest) if dataset_name == "cnn"
                                        else str(config.xsum_manifest)
                ),
                "Max input tokens":     config.max_input_tokens,
                "Max new tokens":       config.max_new_tokens
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


def main():
    args: argparse.Namespace = parse_args()
    config: ExperimentalConfig = ExperimentalConfig()

    print("=" * TEXT_LEN)
    print("QUANTIZATION SUMMARIZATION BENCHMARK")
    print("=" * TEXT_LEN)
    print(f"Samples per dataset : {CONFIG.sample_size}")
    print(f"Random seed         : {CONFIG.seed}")
    print(f"Max new tokens      : {CONFIG.max_new_tokens}")

    cnn_dataset, xsum_dataset = load_datasets(config)

    run_model(cnn_dataset, "cnn", args.model, config)
    run_model(xsum_dataset, "xsum", args.model, config)


if __name__ == "__main__":
    main()
