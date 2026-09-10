from __future__ import annotations

from typing import Any

import torch

from datasets       import Dataset
from transformers   import AutoModelForCausalLM, AutoTokenizer

from transformers.tokenization_utils_base           import BatchEncoding
from transformers.models.qwen2.tokenization_qwen2   import Qwen2Tokenizer
from transformers.models.qwen2.modeling_qwen2       import Qwen2ForCausalLM

from src.model_method import ModelPrecision


MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"


class ModelFP16(ModelPrecision):
    def __init__(self):
        super().__init__("FP16 Baseline")


    def capture_inference_memory(self, observed_peak_mb: float = 0.0):
        """
        Capture inference time memory.

        observed_peak_mb (float): observed peak memory (in Mbis).
        """
        self.inference_rss_mb = self.get_processed_memory()

        if observed_peak_mb > 0.0:
            self.peak_device_memory_mb = observed_peak_mb
        elif torch.backends.mps.is_available():
            self.peak_device_memory_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)


    def run(self, dataset: Dataset, dataset_name: str, model_id: str = MODEL_ID) -> dict[str, Any]:
        self.reset()
        mps_peak_memory_mb: float = 0.0
        print(f"> Running '{self.model_name}' on {len(dataset)} samples.")

        documents:  list[str]
        references: list[str]
        documents, references = self.get_doc_ref(dataset)

        device: str
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"\tTarget device: {device}")

        tokenizer: Qwen2Tokenizer   = AutoTokenizer.from_pretrained(model_id)
        model: Qwen2ForCausalLM     = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype = torch.float16
        ).to(device)

        model.eval()

        observed_peak_mps_mb = 0.0
        if device == "mps":
            observed_peak_mps_mb = max(observed_peak_mps_mb, torch.mps.driver_allocated_memory() / (1024 * 1024))

        first_prompt: str = self.get_prompt_messages(
            documents[0], 
            tokenizer, 
            dataset_name = dataset_name
        )
        first_inp: BatchEncoding = tokenizer(
            first_prompt,
            return_tensors  = "pt",
            truncation      = True,
            max_length      = self.max_inp_tokens
        ).to(device)

        with torch.inference_mode():
            _ = model.generate(
                **first_inp,
                max_new_tokens  = min(self.max_tokens, 8),
                do_sample       = False,
                pad_token_id    = tokenizer.eos_token_id
            )
        print("\tFirst inference completed")

        if device == "mps":
            torch.mps.synchronize()

            observed_peak_mps_mb = max(observed_peak_mps_mb, torch.mps.driver_allocated_memory() / (1024 * 1024))
        self.start_timer()

        for idx, doc in enumerate(documents):
            prompt: str = self.get_prompt_messages(
                doc,
                tokenizer,
                dataset_name = dataset_name
            )

            raw_input_tokens: list[int] = tokenizer.encode(prompt)
            inputs: BatchEncoding = tokenizer(
                prompt,
                return_tensors  = "pt",
                truncation      = True,
                max_length      = self.max_inp_tokens
            ).to(device)

            actual_input_tokens: int = int(inputs.input_ids.shape[1])

            self.record_input_tokens(raw_length = len(raw_input_tokens), actual_length = actual_input_tokens)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens  = self.max_tokens,
                    do_sample       = False,
                    pad_token_id    = tokenizer.eos_token_id
                )

            if device == "mps":
                torch.mps.synchronize()
                observed_peak_mps_mb    = max(observed_peak_mps_mb, torch.mps.driver_allocated_memory() / (1024 * 1024))
                current_mps_memory_mb   = torch.mps.driver_allocated_memory() / (1024 * 1024)
                mps_peak_memory_mb      = max(mps_peak_memory_mb, current_mps_memory_mb)

            generated_tokens = outputs[0, inputs.input_ids.shape[1]:]

            summary = tokenizer.decode(
                generated_tokens,
                skip_special_tokens = True
            ).strip()

            self.predictions.append(summary)

            self.tot_tokens += int(generated_tokens.shape[0])

            self.print_progress(idx, len(dataset))

        if device == "mps":
            torch.mps.synchronize()
            
        self.stop_timer()

        if device == "mps":
            self.peak_device_memory_mb = observed_peak_mps_mb
        self.capture_inference_memory(mps_peak_memory_mb)

        results = self.compute_metrics(dataset, dataset_name, references)

        del model
        del tokenizer

        if device == "mps":
            torch.mps.empty_cache()

        return results