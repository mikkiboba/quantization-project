from __future__ import annotations

from typing import Any

import torch

from datasets       import Dataset
from transformers   import AutoTokenizer, AutoModelForCausalLM

from transformers.models.qwen2.modeling_qwen2       import Qwen2ForCausalLM
from transformers.models.qwen2.tokenization_qwen2   import Qwen2Tokenizer

from gptqmodel import GPTQModel
from gptqmodel import BACKEND

from src.model_method import ModelPrecision


MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"


class ModelGPTQ(ModelPrecision):
    def __init__(self):
        super().__init__("GPTQ 4-bit")


    def run(self, dataset: Dataset, dataset_name: str, model_id: str = MODEL_ID) -> dict[str, Any]:
        self.reset()

        print(f"> Running '{self.model_name}' on {len(dataset)} samples.")

        documents:  list[str]
        references: list[str]
        documents, references = self.get_doc_ref(dataset)

        print(f"\tTarget device: Apple Silicon / GPTQModel")

        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"\tTorch device: {device}")

        tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(model_id)

        model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map = device
        )
        model.eval()

        print(f"\tModel class: {type(model).__name__}")
        print(f"\tQuantization config: {getattr(model, "quantization_config", None)}")

        print(f"\tModel loaded.")

        first_prompt = self.get_prompt_messages(documents[0], tokenizer, dataset_name = dataset_name)

        first_inputs = tokenizer(
            first_prompt,
            return_tensors = "pt",
            truncation = True,
            max_length = self.max_inp_tokens
        )

        first_inputs = {
            key: value.to(device)
            for key, value in first_inputs.items()
        }

        with torch.inference_mode():
            _ = model.generate(
                **first_inputs,
                max_new_tokens  = min(self.max_tokens, 8),
                do_sample       = False,
                pad_token_id    = tokenizer.eos_token_id
            )

        if device == "mps":
            torch.mps.synchronize()

        print("\tFirst inference completed.")

        if device == "mps":
            torch.mps.synchronize()

        self.start_timer()

        for idx, doc in enumerate(documents):
            prompt = self.get_prompt_messages(doc, tokenizer, dataset_name = dataset_name)

            inputs = tokenizer(
                prompt,
                return_tensors  = "pt",
                truncation      = True,
                max_length      = self.max_inp_tokens
            )
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            self.tot_inp_tokens += int(inputs["input_ids"].shape[1])

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens  = self.max_tokens,
                    do_sample       = False,
                    pad_token_id    = tokenizer.eos_token_id
                )

            generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]

            summary = tokenizer.decode(generated_tokens, skip_special_tokens = True).strip()

            self.predictions.append(summary)

            self.tot_tokens += int(generated_tokens.shape[0])

            self.print_progress(idx, len(dataset))

        if device == "mps":
            torch.mps.synchronize()

        self.stop_timer()

        results = self.compute_metrics(dataset, dataset_name, references)

        del model
        del tokenizer

        if device == "mps":
            torch.mps.empty_cache()

        return results




