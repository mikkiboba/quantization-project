from src.model_method   import ModelPrecision
from datasets           import Dataset
from llama_cpp          import Llama
from transformers       import AutoTokenizer

from transformers.tokenization_utils_base           import BatchEncoding
from transformers.models.qwen2.tokenization_qwen2   import Qwen2Tokenizer
from transformers.models.qwen2.modeling_qwen2       import Qwen2ForCausalLM


class ModelGGUF(ModelPrecision):
    def __init__(self):
        super().__init__("GGUF 4-bit (llama.cpp)")


    def run(self, dataset: Dataset, dataset_name: str, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"> Running '{self._model_name}' on {len(dataset)} samples.")

        documents: list[str]
        references: list[str]
        documents, references = self.get_doc_ref(dataset)

        print(f"\t- Target device: Apple Silicon (GPU Metal).")

        tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(model_id)

        llm: Llama = Llama.from_pretrained(
            repo_id      = "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            filename     = "*q4_k_m.gguf",
            n_gpu_layers = -1,
            n_ctx        = 4096,
            verbose      = False
        )

        self.checkpoint_eval(0)

        for idx, doc in enumerate(documents):
            prompt: str = self.get_prompt_messages(doc, tokenizer, dataset_name=dataset_name)

            output = llm(
                prompt      = prompt,
                max_tokens  = self.max_tokens,
                temperature = 0.0,
                echo        = False
            )

            summary: str = output['choices'][0]['text'].strip()
            self.predictions.append(summary)

            self.tot_tokens += output['usage']['completion_tokens']

            self.print_progress(idx, len(dataset))

        self.checkpoint_eval(1)

        return self.compute_metrics(dataset, dataset_name, references)

