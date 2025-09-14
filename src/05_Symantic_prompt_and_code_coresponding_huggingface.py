import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import re
from typing import Dict, Any
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from sklearn.model_selection import KFold
import statsmodels.api as sm
import matplotlib

matplotlib.use('Agg')  # use non-GUI backend
warnings.filterwarnings("ignore")


class UMLPromptAnalyzer:
    def __init__(self):
        self.sentiment_classifier = None
        self.embedding_model = None
        self.embedding_tokenizer = None
        self.rouge = Rouge()
        self._initialize_models()

    def _initialize_models(self):
        try:
            self.sentiment_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            print("✅ Sentiment model loaded")
        except Exception as e:
            print(f"⚠️ Sentiment model failed: {e}")

        try:
            self.embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"⚠️ Embedding model failed: {e}")

    def _get_embedding(self, text: str) -> np.ndarray:
        if not text or not self.embedding_model:
            return np.zeros(384)
        tokens = self.embedding_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            output = self.embedding_model(**tokens)
        embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding / np.linalg.norm(embedding)

    def compute_prompt_uml_similarity(self, prompt: str, uml: str) -> float:
        prompt_emb = self._get_embedding(prompt)
        uml_emb = self._get_embedding(uml)
        return float(np.dot(prompt_emb, uml_emb))

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        try:
            if len(text) > 15:
                result = self.sentiment_classifier(text[:512])
                return {
                    "label": result[0]['label'],
                    "score": result[0]['score']
                }
        except Exception as e:
            print(f"⚠️ Sentiment error: {e}")
        return {"label": "UNKNOWN", "score": 0.5}

    def basic_uml_quality(self, uml: str) -> float:
        if not isinstance(uml, str):
            return 0.0
        score = 50
        score += 10 if '@startuml' in uml and '@enduml' in uml else -10
        score += 10 if 'start' in uml and 'stop' in uml else 0
        score += 10 if len(re.findall(r':.+?;', uml)) >= 3 else 0
        return max(0.0, min(100.0, score))

    def compute_bleu(self, prompt: str, uml: str) -> float:
        reference = [prompt.lower().split()]
        candidate = uml.lower().split()
        smoothie = SmoothingFunction().method4
        return sentence_bleu(reference, candidate, smoothing_function=smoothie)

    def compute_rouge_l(self, prompt: str, uml: str) -> float:
        try:
            scores = self.rouge.get_scores(uml, prompt)
            return scores[0]['rouge-l']['f']
        except Exception as e:
            print(f"⚠️ ROUGE-L computation error: {e}")
            return 0.0


def perform_statistical_analysis(df: pd.DataFrame, metric_columns: list):
    summary = []
    analysis = TTestIndPower()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for col in metric_columns:
        values = df[col].dropna()
        if len(values) < 2:
            continue

        mean = np.mean(values)
        std_dev = np.std(values)
        normal_test = stats.shapiro(values)
        z_stat, p_value = stats.ttest_1samp(values, popmean=0.5)
        effect_size = (mean - 0.5) / std_dev if std_dev else 0
        try:
            power = analysis.solve_power(effect_size=effect_size, nobs=len(values), alpha=0.05)
        except:
            power = np.nan

        u_stat, u_p = stats.mannwhitneyu(values, np.full(len(values), 0.5), alternative='two-sided')
        ci = stats.norm.interval(0.95, loc=mean, scale=std_dev / np.sqrt(len(values)))

        summary.append({
            'Metric': col,
            'Mean': round(mean, 4),
            'Std Dev': round(std_dev, 4),
            'Shapiro-Wilk p': round(normal_test.pvalue, 4),
            'Z-stat': round(z_stat, 4),
            'P-value': round(p_value, 4),
            'Effect Size': round(effect_size, 4),
            'Power (1-β)': round(power, 4) if not np.isnan(power) else 'N/A',
            'Mann-Whitney U p': round(u_p, 4),
            '95% CI Lower': round(ci[0], 4),
            '95% CI Upper': round(ci[1], 4)
        })

        sm.qqplot(values, line='s')
        plt.title(f"Q-Q Plot for {col}")
        plt.savefig(f"qqplot_{col}.png")
        plt.close()

    stats_df = pd.DataFrame(summary)
    stats_df.to_csv("report/03_Symantic_promt_code_Analysis_statistics.csv", index=False)
    stats_df.to_latex("report/03_Symantic_promt_code_Analysis_statistics.tex", index=False)
    stats_df.to_excel("report/03_Symantic_promt_code_Analysis_statistics.xlsx", index=False)
    print("📊 Statistical summary saved to .csv, .tex, and .xlsx")
    return stats_df


def process_dataset_with_huggingface(input_path: str, output_path: str = "report/03_Symantic_promt_code_Analysis.csv"):
    df = pd.read_excel(input_path) if input_path.endswith(".xlsx") else pd.read_csv(input_path)

    if 'prompt' not in df.columns:
        print("❌ Missing 'prompt' column.")
        return

    analyzer = UMLPromptAnalyzer()
    uml_columns = [col for col in df.columns if col != 'prompt' and df[col].dtype == 'object']
    all_metric_columns = []

    for uml_col in tqdm(uml_columns, desc="Processing UML Columns"):
        sim_list, match_list, qual_list, bleu_list, rouge_list = [], [], [], [], []

        for _, row in df.iterrows():
            prompt = str(row['prompt'])
            uml_code = str(row.get(uml_col, ''))

            if not uml_code or len(uml_code) < 20:
                sim_list.append(0.0)
                match_list.append(0.0)
                qual_list.append(0.0)
                bleu_list.append(0.0)
                rouge_list.append(0.0)
                continue

            sim = analyzer.compute_prompt_uml_similarity(prompt, uml_code)
            qual = analyzer.basic_uml_quality(uml_code)
            match = round(sim * qual, 4)
            bleu = analyzer.compute_bleu(prompt, uml_code)
            rouge = analyzer.compute_rouge_l(prompt, uml_code)

            sim_list.append(round(sim, 4))
            match_list.append(match)
            qual_list.append(round(qual, 2))
            bleu_list.append(round(bleu, 4))
            rouge_list.append(round(rouge, 4))

        for metric, values in zip(['similarity', 'quality', 'match_rate', 'bleu', 'rougeL'],
                                  [sim_list, qual_list, match_list, bleu_list, rouge_list]):
            col_name = f'prompt_{uml_col}_{metric}'
            df[col_name] = values
            all_metric_columns.append(col_name)

    df.to_csv(output_path, index=False)
    print(f"✅ Results saved to {output_path}")

    stats_df = perform_statistical_analysis(df, all_metric_columns)
    visualize_results(df, all_metric_columns)
    return df, stats_df


def visualize_results(df, metric_columns):
    plt.figure(figsize=(14, 6))
    for col in metric_columns:
        if 'match_rate' in col:
            sns.kdeplot(df[col], label=col)
    plt.title("Match Rate Distributions")
    plt.legend()
    plt.savefig("report/Prompt_match_rate_distribution.png")
    plt.close()


def main():
    input_file = "data/test_dataset2.xlsx"
    output_file = "report/03_Symantic_promt_code_Analysis.csv"
    df, stats_df = process_dataset_with_huggingface(input_file, output_file)


if __name__ == "__main__":
    main()
