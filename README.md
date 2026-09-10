# Quantization analysis on abstractive summarization
This is a project for the Natural Language Processing (NLP) course. It aims to study and analyze the differences of different quantization methods and their impact on the task of abstractive summarization.

## How to run the code
The current repository is missing the dataset manifests. After installing the necessary libraries from the file `requirements.txt`, you must run the file `src/eval_manifest.py`.

```bash
# install libraries in requirements.txt
pip install -r requirements.txt
```
```bash
# generate manifests
python -m src/eval_manifest
```

After this, the main code can be run with different arguments, for example:
```bash
python -m main --model fp16 --limit 500 --dataset xsum
```

**Note**: the generated manifests have a maximum of 500 samples. Change it in the code to have more of them.
