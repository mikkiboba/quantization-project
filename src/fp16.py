from src.model_method import ModelPrecision
from datasets         import Dataset
from transformers     import AutoModelForCausalLM, AutoTokenizer

from transformers.tokenization_utils_base           import BatchEncoding
from transformers.models.qwen2.tokenization_qwen2   import Qwen2Tokenizer
from transformers.models.qwen2.modeling_qwen2       import Qwen2ForCausalLM

import torch


class ModelFP16(ModelPrecision):
    def __init__(self):
        super().__init__("FP16 Baseline")


    def run(self, dataset: Dataset, dataset_name: str, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"> Running '{self._model_name}' on {len(dataset)} samples.")

        documents: list[str]
        references: list[str]
        documents, references = self.get_doc_ref(dataset)

        device: str = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"\t- Target device: {device}")

        tokenizer: Qwen2Tokenizer
        model: Qwen2ForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype = torch.float16
        ).to(device)

        self.checkpoint_eval(0)

        for idx, doc in enumerate(documents):
            prompt: str = self.get_prompt_messages(doc, tokenizer, dataset_name=dataset_name)
            inputs: BatchEncoding = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

            with torch.no_grad():
                outputs: torch.Tensor = model.generate(
                    **inputs,
                    max_new_tokens  = self.max_tokens,
                    do_sample       = False,
                    pad_token_id    = tokenizer.eos_token_id
                )

            gen_tokens: torch.Tensor = outputs[0][inputs.input_ids.shape[1]:]

            summary: str = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            self.predictions.append(summary)
            self.tot_tokens += len(gen_tokens)

            self.print_progress(idx, len(dataset))

        self.checkpoint_eval(1)

        return self.compute_metrics(dataset, dataset_name, references)