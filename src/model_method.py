from __future__ import annotations

import time
import psutil

from abc        import ABC, abstractmethod
from pathlib    import Path
from typing     import Any
from datasets   import Dataset

import src.evaluator as evaluator

from src.db_loader import DatabaseError


BYTES_TO_MB = 1048576 # 1024 * 1024


class ModelPrecision(ABC):
    """
    Shared base class for any model configuration.

    A subclass has to:
        1. Load its model.
        2. Generate summaries.
        3. Measure inference time.
        4. Make predictions.
    """

    # * attributes
    model_name: str
    _process:   psutil.Process

    _start_time: int
    _end_time:   int

    peak_mem: float

    predictions: list[str]

    tot_tokens:     int
    tot_inp_tokens: int 

    max_tokens:     int
    max_inp_tokens: int

    elapsed_seconds:    float
    memory_mb:          float


    def __init__(self, model_name: str):
        self.model_name = model_name
        self._process = psutil.Process()

        self.predictions = []

        self.tot_tokens         = 0
        self.total_input_tokens = 0

        self.max_inp_tokens = 1024
        self.max_tokens     = 128

        self.elapsed_seconds    = 0.0
        self.memory_mb          = 0.0


    def reset(self):
        """
        Reset model values.
        """

        self.predictions        = []
        self.tot_tokens         = 0
        self.tot_inp_tokens     = 0
        self.elapsed_seconds    = 0.0
        self.memory_mb          = 0.0


    def get_doc_ref(self, dataset: Dataset) -> tuple[list[str], list[str]]:
        """
        Receive the tuple containing documents and reference summaries from the specified dataset.
        
        Note: The dataset MUST have columns named "document" and "summary".
        """

        documents:  list[str]
        references: list[str]

        try:
            documents   = dataset["document"]
            references  = dataset["summary"]
        except Exception as e:
            raise DatabaseError("Dataset must contain 'document' and 'summary' columns") from e

        return documents, references


    # ! rss and not 'vram' or 'peak memory'
    def get_processed_memory(self) -> float:
        """
        Return current python RSS (in mb).
        """

        return self._process.memory_info().rss / BYTES_TO_MB


    def get_prompt_messages(
            self,
            document: str, 
            tokenizer: Any, 
            *, 
            dataset_name: str, 
            add_generation_prompt: bool = True
        ) -> str:
        """
        Return the prompt otbtained from a chat template.
        """

        instruction: str
        if dataset_name.lower() == "xsum":
            instruction = "Summarize the following document in one concise sentence:"
        else:
            instruction = "Summarize the following document in 2 or 3 concise sentences:"

        messages: list[dict[str, str]] = [
            {
                "role":     "system",
                "content":  "You are a precise summarization assistant."
            },
            {
                "role":     "user",
                "content":  f"{instruction}\n\n{document}",
            },
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = add_generation_prompt
        )


    def start_timer(self):
        """
        Start inference timer.
        """

        self._start_time = time.perf_counter()


    def stop_timer(self):
        """
        Stop inference timer.
        """

        self.elapsed_seconds = time.perf_counter() - self._start_time


    def print_progress(self, idx: int, dataset_len: int):
        """
        Print benchmark progress.
        """

        if (idx + 1) % 5 == 0 or (idx + 1) == dataset_len:
            print(f"\t\tProcessed [{idx + 1}/{dataset_len}] samples.")


    def compute_metrics(self, dataset: Dataset, dataset_name: str, references: list[str]) -> dict[str, Any]:
        """
        Compute quality and efficiency metrics. 
        Return a dictionary of all the needed metrics for the evaluation of the model.
        """

        if len(self.predictions) != len(references):
            raise RuntimeError(
                f"Predictions/references mismatch: {len(self.predictions)} predictions, "
                "{len(references)} references."
            )

        metric_evaluator: evaluator.MetricEvaluator = evaluator.MetricEvaluator()

        metrics: dict = metric_evaluator.compute(
            predictions = self.predictions,
            references  = references
        )

        avg_out_tokens: float = (
            self.tot_tokens / len(self.predictions) if self.predictions
            else 0.0
        )

        avg_inp_tokens: float = (
            self.tot_inp_tokens / len(self.predictions) if self.predictions
            else 0.0
        )

        speed: float = (
            self.tot_tokens / self.elapsed_seconds if self.elapsed_seconds > 0
            else 0.0
        )

        return {
            "Model":    self.model_name,
            "Dataset":  dataset_name,
            "Samples":  len(dataset),

            **metrics,

            "Latency (s)":              round(self.elapsed_seconds, evaluator.ROUNDING),
            "Speed (token/s)":          round(speed, evaluator.ROUNDING),
            "Generated tokens":         self.tot_tokens,
            "Average output tokens":    round(avg_out_tokens, evaluator.ROUNDING),
            "Average input tokens":     round(avg_inp_tokens, evaluator.ROUNDING),
            "RSS (MB)":                 round(self.get_processed_memory(), evaluator.ROUNDING)
        }


    @abstractmethod
    def run(self, dataset: Dataset, dataset_name: str) -> dict[str, Any]:
        """
        Run the model and return benchmark metrics
        """
        raise NotImplementedError

    