import evaluate


ROUNDING: int = 3


class MetricEvaluator:
    def __init__(self):
        print("> Loading evaluation metrics (ROUGE and BERTScore).")    
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")


    def compute(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        r_scores = self.rouge.compute(predictions=predictions, references=references)
        b_scores = self.bertscore.compute(predictions=predictions, references=references, lang="en")

        avg_bert_f1 = sum(b_scores["f1"]) / len(b_scores["f1"])

        return {
            "ROUGE-1": round(r_scores["rouge1"] * 100, ROUNDING),
            "ROUGE-2": round(r_scores["rouge2"] * 100, ROUNDING),
            "ROUGE-L": round(r_scores["rougeL"] * 100, ROUNDING),
            "BERTScore-F1": round(avg_bert_f1 * 100, ROUNDING)
        }