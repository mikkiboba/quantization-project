from __future__ import annotations

from typing         import Any
from datasets       import Dataset
from llama_cpp      import Llama
from transformers   import AutoTokenizer

from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

from src.model_method import ModelPrecision


MODEL_ID: str       = "Qwen/Qwen2.5-1.5B-Instruct"
GGUF_REPO_ID: str   = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_FILENAME: str  = "*q4_k_m.gguf"

MAX_TOKENS_LIMIT: int = 8


class ModelGGUF(ModelPrecision):
    def __init__(self):
        super().__init__("GGUF Q4_K_M (llama.cpp)")


    def run(self, dataset: Dataset, dataset_name: str, model_id: str = MODEL_ID) -> dict[str, Any]:
        self.reset()

        print(f"> Running '{self.model_name}' on {len(dataset)} samples.")

        documents:  list[str]
        references: list[str]
        documents, references = self.get_doc_ref(dataset)

        # ? non posso cambiare device per questo tipo di quantization
        print("Target device Apple Silicon (llama.cpp + Metal)")

        tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("\tLoading Q4_K_M GGUF model...")
        llm: Llama = Llama.from_pretrained(
            repo_id         = GGUF_REPO_ID,
            filename        = GGUF_FILENAME,
            n_gpu_layers    = -1,
            n_ctx           = 4096,
            verbose         = False
        )
        print("\tModel loaded.")

        first_prompt: str = self.get_prompt_messages(documents[0], tokenizer, dataset_name = dataset_name)
        print("\tRunning first inference...")
        _ = llm(
            prompt      = first_prompt,
            max_tokens  = min(self.max_tokens, MAX_TOKENS_LIMIT),
            temperature = 0.0,
            top_p       = 1.0,
            echo        = False
        )
        print("\tFirst inference completed.")

        self.start_timer()

        for idx, doc in enumerate(documents):
            prompt: str = self.get_prompt_messages(doc, tokenizer, dataset_name = dataset_name)

            raw_inp_tokens: list[int] = llm.tokenize(
                prompt.encode("utf-8"),
                add_bos = True,
                special = True
            )

            actual_inp_tokens: list[int] = (
                raw_inp_tokens[: self.max_inp_tokens]
            )


            self.record_input_tokens(len(raw_inp_tokens), len(actual_inp_tokens))

            output = llm(
                prompt      = actual_inp_tokens,
                max_tokens  = self.max_tokens,
                temperature = 0.0,
                top_p       = 1.0,
                echo        = False
            )

            summary: str = output["choices"][0]["text"].strip()

            self.predictions.append(summary)

            completion_tokens = int(output["usage"]["completion_tokens"])
            self.tot_tokens += completion_tokens

            self.print_progress(idx, len(dataset))

        self.stop_timer()

        self.capture_inference_memory()

        results = self.compute_metrics(dataset, dataset_name, references)

        del llm
        del tokenizer

        return results 