from __future__ import annotations

import re
import time
import psutil

from abc        import ABC, abstractmethod
from typing     import Any
from datasets   import Dataset

import src.evaluator as evaluator

from src.db_loader import DatabaseError

# 1024 * 1024
BYTES_TO_MB = 1024 * 1024 


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

    _process:       psutil.Process
    _start_time:    float

    predictions: list[str]

    tot_tokens:         int
    tot_inp_tokens:     int 
    tot_raw_inp_tokens: int

    max_tokens:     int
    max_inp_tokens: int

    truncated_examples: int

    elapsed_seconds:        float
    peak_device_memory_mb:  float
    inference_rss_mb:       float


    def __init__(self, model_name: str):
        self.model_name         = model_name
        self._process           = psutil.Process()

        self.predictions        = []

        self.tot_tokens         = 0
        self.tot_inp_tokens     = 0
        self.tot_raw_inp_tokens = 0

        self.max_inp_tokens     = 1024
        self.max_tokens         = 128

        self.truncated_examples = 0

        self.elapsed_seconds        = 0.0
        self.peak_device_memory_mb  = 0.0
        self.inference_rss_mb       = 0.0        


    def reset(self):
        """
        Reset model values.
        """

        self.predictions        = []
        self.tot_tokens         = 0
        self.tot_inp_tokens     = 0
        self.tot_raw_inp_tokens = 0
        self.truncated_examples = 0
        self.elapsed_seconds    = 0.0
        self.peak_device_memory_mb = 0.0
        self.inference_rss_mb   = 0.0


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


    def record_input_tokens(self, raw_length: int, actual_length: int):
        self.tot_raw_inp_tokens += raw_length
        self.tot_inp_tokens     += actual_length

        if raw_length > self.max_inp_tokens:
            self.truncated_examples += 1


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


    @staticmethod
    def count_sentences(text: str) -> int:
        """
        Estimate the number of sentences in a summary.
        """

        text = text.strip()
        if not text:
            return 0

        sentences = re.split(r"(?<=[.!?])\s+", text)

        return len([sentence for sentence in sentences if sentence.strip()])


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


    def capture_inference_memory(self):
        """
        Capture process RSS after inference and before quality evaluation.
        """
        self.inference_rss_mb = self.get_processed_memory()


    def compute_metrics(self, dataset: Dataset, dataset_name: str, references: list[str]) -> dict[str, Any]:
        """
        Compute quality and efficiency metrics. 
        Return a dictionary of all the needed metrics for the evaluation of the model.
        """

        if len(self.predictions) != len(references):
            raise RuntimeError(
                f"Predictions/references mismatch: {len(self.predictions)} predictions, "
                f"{len(references)} references."
            )

        metric_evaluator: evaluator.MetricEvaluator = evaluator.MetricEvaluator()

        metrics: dict = metric_evaluator.compute(
            predictions = self.predictions,
            references  = references
        )

        n_examples: int = len(self.predictions)
        avg_out_tokens: float = (
            self.tot_tokens / n_examples if n_examples > 0
            else 0.0
        )

        avg_inp_tokens: float = (
            self.tot_inp_tokens / n_examples if n_examples > 0
            else 0.0
        )

        avg_raw_inp_tokens: float = (
            self.tot_raw_inp_tokens / n_examples if n_examples > 0
            else 0.0
        )

        truncation_rate: float = (
            self.truncated_examples / n_examples if n_examples > 0
            else 0.0
        )

        empty_outputs: int = sum(1 for pred in self.predictions if not pred.strip())

        sentence_counts: list[int] = [self.count_sentences(pred) for pred in self.predictions]

        avg_sentences: float = (
            sum(sentence_counts) / n_examples if n_examples > 0
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
            "Average raw input tokens": round(avg_raw_inp_tokens, evaluator.ROUNDING),
            "Truncated examples":       self.truncated_examples,
            "Truncation rate":          round(truncation_rate, evaluator.ROUNDING),
            "Average sentences":        round(avg_sentences, evaluator.ROUNDING),
            "Empty outputs":            empty_outputs,
            "Inference RSS (MiB)":      round(self.inference_rss_mb, evaluator.ROUNDING),
            "Peak backend memory (MiB)": round(self.peak_device_memory_mb, evaluator.ROUNDING)
        }


    @abstractmethod
    def run(self, dataset: Dataset, dataset_name: str) -> dict[str, Any]:
        """
        Run the model and return benchmark metrics
        """
        raise NotImplementedError

    