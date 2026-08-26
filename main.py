from src.db_loader    import load
from src.model_method import ModelPrecision
from datasets         import Dataset

import src.fp16 as fp16
import src.mlx  as mlx
import src.gguf as gguf
import src.bnb  as bnb

import pandas as pd
import os
import transformers
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()


SAMPLE_SIZE: int = 200
TEXT_REPEAT: int = 60
OUTPUT_DIR: str  = "results/benchmark_results_"
SAMPLES_DIR: str = "data/qualitative_examples_"

SAMPLES_DIR_XSUM: str = "data/qualitative_examples_xsum.csv"
SAMPLES_DIR_CNN: str  = "data/qualitative_examples_cnn.csv"


def load_datasets(num_samples: int | None = None) -> tuple[Dataset, Dataset]:
    cnn_ds = load(
        db_name     = "abisee/cnn_dailymail",
        db_version  = "3.0.0",
        db_split    = "test",
        num_samples = num_samples
    )

    xsum_ds = load(
        db_name     = "EdinburghNLP/xsum",
        db_split    = "test",
        num_samples = num_samples
    )

    cnn_ds = cnn_ds.rename_column("article", "document")
    cnn_ds = cnn_ds.rename_column("highlights", "summary")

    return cnn_ds, xsum_ds


def load_models() -> list[ModelPrecision]:
    models: list[ModelPrecision] = []
    models.append(fp16.ModelFP16())
    models.append(mlx.ModelMLX())
    models.append(gguf.ModelGGUF())
    models.append(bnb.ModelBNB())

    return models


def run_models_on_dataset(dataset: Dataset, dataset_name: str):
    dataset_sample: Dataset      = dataset.select(range(SAMPLE_SIZE))
    models: list[ModelPrecision] = load_models()

    all_results: list[dict] = []
    for model in models:
        all_results.append(model.run(dataset_sample, dataset_name))

    result_dir: str = f"{OUTPUT_DIR}{dataset_name}.csv"
    df_results: pd.DataFrame = pd.DataFrame(all_results)
    df_results.to_csv(result_dir, index=False)
    print(f"✔ Results saved in {result_dir}")

    df_examples: pd.DataFrame = pd.DataFrame({
        "Document":             dataset["document"],
        "Human-made Reference": dataset["summary"]
    })

    for model in models:
        if len(model.predictions) == len(df_examples):
            df_examples[model._model_name] = model.predictions

    examples_dir: str = f"{SAMPLES_DIR}{dataset_name}.csv"
    df_examples.to_csv(examples_dir, index=False)
    print(f"✔ Examples saved in {examples_dir}.")

    


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    cnn_ds:  Dataset
    xsum_ds: Dataset
    cnn_ds, xsum_ds = load_datasets()

    cnn_sample:  Dataset = cnn_ds.select(range(SAMPLE_SIZE))
    xsum_sample: Dataset = xsum_ds.select(range(SAMPLE_SIZE))

    run_models_on_dataset(cnn_sample, "cnn")
    run_models_on_dataset(xsum_sample, "xsum")

if __name__ == "__main__":
    main()

    