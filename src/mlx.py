from src.model_method    import ModelPrecision
from datasets            import Dataset
from mlx_lm              import load as mlx_load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

from mlx_lm.models.qwen2    import Model
from mlx_lm.tokenizer_utils import TokenizerWrapper


class ModelMLX(ModelPrecision):
    def __init__(self):
        super().__init__("MLX 4-bit")


    def run(self, dataset: Dataset, dataset_name: str, model_id: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"):
        print(f"> Running '{self._model_name}' on {len(dataset)} samples.")

        documents: list[str]
        references: list[str]
        documents, references = self.get_doc_ref(dataset)

        print("\t- Target device: Apple Silicon (UM).")

        model: Model
        tokenizer: TokenizerWrapper
        model, tokenizer = mlx_load(model_id)

        self.checkpoint_eval(0)

        sampler: function = make_sampler(temp=0.0)
        for idx, doc in enumerate(documents):
            prompt: str = self.get_prompt_messages(doc, tokenizer, dataset_name=dataset_name)

            output: str = mlx_generate(
                model       = model,
                tokenizer   = tokenizer,
                prompt      = prompt,
                max_tokens  = self.max_tokens,
                sampler     = sampler,
                verbose     = False
            )

            summary: str = output.strip()
            self.predictions.append(summary)

            self.tot_tokens += len(tokenizer.encode(summary))

            self.print_progress(idx, len(dataset))

        self.checkpoint_eval(1)

        return self.compute_metrics(dataset, dataset_name, references)