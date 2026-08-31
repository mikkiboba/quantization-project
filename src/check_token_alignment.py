from __future__ import annotations

from mlx_lm import load as mlx_load
from transformers import AutoTokenizer
from llama_cpp import Llama


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

MLX_MODEL_ID = (
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
)

GGUF_REPO_ID = (
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
)

GGUF_FILENAME = "*q4_k_m.gguf"


DOCUMENT = (
    "The company announced today that it will open "
    "a new research center next year."
)


def build_prompt(tokenizer) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a precise summarization assistant.",
        },
        {
            "role": "user",
            "content": (
                "Summarize the following document "
                "in 2 or 3 concise sentences:\n\n"
                f"{DOCUMENT}"
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:

    # ------------------------------------------------------------
    # Hugging Face / FP16
    # ------------------------------------------------------------

    hf_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    prompt_hf = build_prompt(hf_tokenizer)

    hf_tokens = hf_tokenizer.encode(
        prompt_hf
    )

    # ------------------------------------------------------------
    # MLX
    # ------------------------------------------------------------

    _, mlx_tokenizer = mlx_load(
        MLX_MODEL_ID
    )

    prompt_mlx = build_prompt(
        mlx_tokenizer
    )

    mlx_tokens = mlx_tokenizer.encode(
        prompt_mlx
    )

    # ------------------------------------------------------------
    # GGUF
    # ------------------------------------------------------------

    gguf = Llama.from_pretrained(
        repo_id=GGUF_REPO_ID,
        filename=GGUF_FILENAME,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
    )

    prompt_gguf = build_prompt(
        hf_tokenizer
    )

    gguf_tokens = gguf.tokenize(
        prompt_gguf.encode("utf-8"),
        add_bos=True,
        special=True,
    )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print()
    print("PROMPT EQUALITY")
    print("=" * 60)

    print(
        "HF == MLX prompt:",
        prompt_hf == prompt_mlx,
    )

    print()
    print("TOKEN LENGTHS")
    print("=" * 60)

    print(
        "HF:",
        len(hf_tokens),
    )

    print(
        "MLX:",
        len(mlx_tokens),
    )

    print(
        "GGUF:",
        len(gguf_tokens),
    )

    print()
    print("TOKEN ID EQUALITY")
    print("=" * 60)

    print(
        "HF == MLX:",
        hf_tokens == mlx_tokens,
    )

    # GGUF may include BOS independently.
    # Compare the common length first.
    common_length = min(
        len(hf_tokens),
        len(gguf_tokens),
    )

    print(
        "HF == GGUF:",
        hf_tokens[:common_length]
        == gguf_tokens[:common_length],
    )

    print()
    print("FIRST 30 TOKENS")
    print("=" * 60)

    print("HF:")
    print(hf_tokens[:30])

    print()
    print("MLX:")
    print(mlx_tokens[:30])

    print()
    print("GGUF:")
    print(gguf_tokens[:30])


if __name__ == "__main__":
    main()