from __future__ import annotations

from typing     import Any
from datasets   import Dataset

import mlx.core as mx
from mlx_lm                 import generate as mlx_generate
from mlx_lm                 import load     as mlx_load
from mlx_lm.sample_utils    import make_sampler
from mlx_lm.models.qwen2    import Model
from mlx_lm.tokenizer_utils import TokenizerWrapper

from src.model_method import ModelPrecision


MAX_TOKENS_LIMIT: int = 8

MODEL_ID: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


class ModelMLX(ModelPrecision):
    def __init__(self):
        super().__init__("MLX 4-bit")


    def capture_inference_memory(self):
        """
        Capture MLX memory and process RSS after inference.
        """

        self.inference_rss_mb = self.get_processed_memory()
        self.peak_device_memory_mb = mx.get_peak_memory() / (1024 * 1024)

    def run(
            self,
            dataset: Dataset,
            dataset_name: str,
            model_id: str = MODEL_ID
    ) -> dict[str, Any]:

        self.reset()

        print(f"Running '{self.model_name}' on {len(dataset)} samples.")

        documents:  list[str]
        references: list[str]
        documents, references = self.get_doc_ref(dataset)

        print("\tTarget device: Apple Silicon / MLX.") # it's mandatory here to be apple

        model: Model
        tokenizer: TokenizerWrapper
        model, tokenizer = mlx_load(model_id)

        sampler: function = make_sampler(temp = 0.0)

        first_prompt: str = self.get_prompt_messages(
            documents[0], 
            tokenizer, 
            dataset_name = dataset_name
        )

        print("\tRunning first inference.")

        _ = mlx_generate(
            model       = model,
            tokenizer   = tokenizer,
            prompt      = first_prompt,
            max_tokens  = min(self.max_tokens, MAX_TOKENS_LIMIT),
            sampler     = sampler,
            verbose     = False
        )

        mx.eval(model)

        print("\tFirst inference completed.")     

        self.start_timer()

        for idx, doc in enumerate(documents):
            prompt: str = self.get_prompt_messages(
                doc,
                tokenizer,
                dataset_name = dataset_name
            )

            raw_inp_tokens: list[int]       = tokenizer.encode(prompt)
            actual_inp_tokens: list[int]    = raw_inp_tokens[:self.max_inp_tokens]

            self.record_input_tokens(raw_length = len(raw_inp_tokens), actual_length = len(actual_inp_tokens))

            output: str = mlx_generate(
                model       = model,
                tokenizer   = tokenizer,
                prompt      = actual_inp_tokens,
                max_tokens  = self.max_tokens,
                sampler     = sampler,
                verbose     = False
            )
            summary: str = output.strip()

            self.predictions.append(summary)

            generated_tokens = tokenizer.encode(summary)
            self.tot_tokens += len(generated_tokens)

            self.print_progress(idx, len(dataset))

        mx.eval(model)

        self.stop_timer()

        self.capture_inference_memory()

        results: dict[str, Any] = self.compute_metrics(dataset, dataset_name, references)

        return results

