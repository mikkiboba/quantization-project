from __future__ import annotations

import warnings
import pandas as pd
import transformers
import argparse

from dataclasses    import dataclass
from pathlib        import Path
from datasets       import Dataset
from typing         import Any

from src.db_loader      import load_evaluation_manifest
from src.model_method   import ModelPrecision
from src.environment    import get_environment_info

import src.model_fp16 as fp16
import src.model_gguf as gguf
import src.model_mlx  as mlx
import src.model_gptq as gptq


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

    seed: int               = 42

    output_dir: Path        = Path("results")
    samples_dir: Path       = Path("data")

    max_new_tokens: int     = 128
    max_input_tokens: int   = 1024

    cnn_manifest: Path      = Path("data/cnn_eval_seed42_500.csv")
    xsum_manifest: Path     = Path("data/xsum_eval_seed42_500.csv")


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

    parser.add_argument(
        "--limit",
        type = int,
        default = None,
        help = "Temporarily limit the number of evaluation examples used."
    )

    parser.add_argument(
        "--dataset",
        required = True,
        choices = [
            "cnn",
            "xsum"
        ],
        help = "Dataset for the benchmark."
    )

    return parser.parse_args()


def load_datasets(config: ExperimentalConfig) -> tuple[Dataset, Dataset]:
    """
    Load the manifest.
    """

    cnn: Dataset    = load_evaluation_manifest(config.cnn_manifest)
    xsum: Dataset   = load_evaluation_manifest(config.xsum_manifest)

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
        return gguf.ModelGGUF()
    if model_name == "gptq":
        return gptq.ModelGPTQ()

    raise ValueError(f"Unknown model: {model_name}")


def save_examples(
        dataset: Dataset,
        dataset_name: str,
        model: ModelPrecision,
        model_name: str,
        config: ExperimentalConfig,
):
    """
    Save documents and model predictions.
    """

    if len(model.predictions) != len(dataset):
        raise RuntimeError(f"Prediction count mismatch for model '{model.model_name}': {len(model.predictions)} predictions for {len(dataset)} samples.")
    
    rows = {
        "sample_id":            dataset["sample_id"],
        "Document":             dataset["document"],
        "Human-made Reference": dataset["summary"],
        model.model_name:       model.predictions,
    }

    examples: pd.DataFrame = pd.DataFrame(rows)

    config.samples_dir.mkdir(parents=True, exist_ok=True)

    output_path: Path = (
        config.samples_dir
        / f"qualitative_{model_name}_{dataset_name}.csv"
    )

    examples.to_csv(output_path, index=False)

    print(f"✓ Samples saved in {output_path}")


def run_model(
        dataset: Dataset,
        dataset_name: str,
        model_name: str,
        config: ExperimentalConfig,
        environment: dict[str, Any]
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
            "Seed": config.seed,

            "Evaluation manifest": (
                str(config.cnn_manifest)
                if dataset_name == "cnn"
                else str(config.xsum_manifest)
            ),

            "Max input tokens": (
                config.max_input_tokens
            ),

            "Max new tokens": (
                config.max_new_tokens
            ),

            **environment,
        }
    )

    config.output_dir.mkdir(parents = True, exist_ok = True)

    benchmark_path: Path = config.output_dir / f"benchmark_{model_name}_{dataset_name}.csv"

    pd.DataFrame([result]).to_csv(benchmark_path, index = False)

    print()
    print(f"✓ Results saved to: {benchmark_path}")

    save_examples(dataset, dataset_name, model, model_name, config)


def main():
    args: argparse.Namespace = parse_args()
    config: ExperimentalConfig = ExperimentalConfig()
    environment = get_environment_info()

    print("=" * TEXT_LEN)
    print("QUANTIZATION SUMMARIZATION BENCHMARK")
    print("=" * TEXT_LEN)

    print(f"Model:                  {args.model}")
    print(f"Random seed:            {config.seed}")
    print(f"Max input tokens:       {config.max_input_tokens}")
    print(f"Max new tokens:         {config.max_new_tokens}")
    print(f"CNN/Dailymail manifest: {config.cnn_manifest}")
    print(f"XSum manifest:          {config.xsum_manifest}")

    dataset = load_evaluation_manifest(
        config.cnn_manifest if args.dataset == "cnn"
        else config.xsum_manifest
    )

    if args.limit is not None:
        dataset = dataset.select(
            range(
                min(args.limit, len(dataset))
            )
        )

    run_model(
        dataset         = dataset,
        dataset_name    = args.dataset,
        model_name      = args.model,
        config          = config,
        environment     = environment
    )


if __name__ == "__main__":
    main()
