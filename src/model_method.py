from src.db_loader import DatabaseNotFoundError
from datasets import Dataset
import psutil
import time
import src.evaluator as evaluator


# from bytes to megabytes
CONVERSION_FACTOR: int = 1024 * 1024 


class ModelPrecision():

    _model_name: str
    _process: psutil.Process

    start_time: int
    end_time:   int

    peak_mem: float

    predictions:    list[str]
    tot_tokens:     int 

    max_tokens: int 


    def __init__(self, model_name: str):
        """Define a precision model and its evaluation values."""
        self._process       = psutil.Process()
        self.predictions    = []
        self.tot_tokens     = 0

        self._model_name = model_name

        self.max_tokens = 128


    def get_doc_ref(self, dataset: Dataset) -> tuple[list[str], list[str]]:
        """Receive the tuple containing documents and references from the specified database.
        The dataset MUST have columns named "document" and "summary"."""
        documents:  list[str]
        references: list[str]

        try:
            documents  = dataset["document"]
            references = dataset["summary"]
        except Exception:
            raise DatabaseNotFoundError(f"Database not found or error with columns.")

        return documents, references


    def get_peak_memory_mb(self) -> float:
        """Dynamically detect the active framework and pull its peak footprint."""

        if "MLX" in self._model_name:
            import importlib
            mx = importlib.import_module("mlx.core")
            return mx.metal.get_peak_memory() / CONVERSION_FACTOR
        elif "GGUF" in self._model_name:
            return self._process.memory_info().rss / CONVERSION_FACTOR
        else:
            mem: float = 0.0
            import sys
            if "torch" in sys.modules:
                import torch
                if torch.backends.mps.is_available():
                    mem = torch.mps.current_allocated_memory() / CONVERSION_FACTOR
                elif torch.cuda.is_available():
                    mem = torch.cuda.max_memory_allocated() / CONVERSION_FACTOR

            if mem == 0:
                mem = self._process.memory_info().rss / CONVERSION_FACTOR

            return mem


    def checkpoint_eval(self, checkpoint: int = 0):
        """Define memory and time over a checkpoint (0 -> start; 1 -> end)."""
        if checkpoint == 0:
            self.start_time = time.perf_counter()
        elif checkpoint == 1:
            self.end_time = time.perf_counter() - self.start_time
            self.peak_mem = self.get_peak_memory_mb()
        else:
            print("X: invalid checkpoint value (0 or 1).")



    def get_prompt_messages(self, document: str, tokenizer, tokenize: bool = False, dataset_name: str = "xsum", add_generation_prompt: bool = True) -> str:
        """Return the prompt otbtained from a chat template."""
        instructions: str
        if dataset_name == "xsum":
            instructions = "Summarize the following document in one concise sentence:"
        else:
            instructions = "Summarize the following document in 2 or 3 concise sentences:"
            
        messages = [
            {"role": "system", "content": "You are a precise summarization assistant."},
            {"role": "user", "content": f"{instructions}:\n\n{document}"}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
        return prompt


    def print_progress(self, idx: int, db_len: int):
        """Print the progress of the processed samples."""
        if (idx + 1) % 5 == 0 or (idx + 1) == db_len:
            print(f"\t\tProcessed [{idx + 1}/{db_len}] samples.")


    def compute_metrics(self, dataset: Dataset, dataset_name: str, references: list[str]) -> dict:
        """Get a dictionary of all the needed metrics for the evaluation of the model."""
        print("> Evaluation")
        ev: evaluator.MetricEvaluator = evaluator.MetricEvaluator()
        metrics: dict = ev.compute(predictions=self.predictions, references=references)

        results: dict = {
                "Model": self._model_name,
                "Dataset": dataset_name,
                "Samples": len(dataset),
                **metrics,
                "Latency (s)": round(self.end_time, evaluator.ROUNDING),
                "Speed (tok/s)": round(self.tot_tokens / self.end_time if self.end_time > 0 else 0, evaluator.ROUNDING),
                "RAM Usage (MB)": round(max(0, self.peak_mem), evaluator.ROUNDING)
            }
        return results


    def run(self, dataset: Dataset, dataset_name: str, model_id: str):
        """Run the ModelPrecision."""
        print(f"-- The `run()` function is not implemented yet.")


    def reset(self):
        self._process       = psutil.Process()
        self.predictions    = []
        self.tot_tokens     = 0

            
