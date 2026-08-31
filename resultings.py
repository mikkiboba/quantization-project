import pandas as pd

models = ["fp16", "mlx", "gguf"]
datasets = ["cnn", "xsum"]

files = []

for model in models:
    for dataset in datasets:
        path = f"results/benchmark_{model}_{dataset}.csv"
        files.append(path)

dfs = []

for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv("combined.csv", index=False)