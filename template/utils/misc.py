# {% include 'template/license_header' %}

import numpy as np
from datasets import load_metric

from zenml.enums import StrEnum

def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """Compute the metrics for the model.
    
    Args:
        eval_pred: The evaluation prediction.
    
    Returns:
        The metrics for the model.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculate the mertic using the predicted and true value
    accuracy = load_metric("accuracy").compute(predictions=predictions, references=labels)
    f1 = load_metric("f1").compute(predictions=predictions, references=labels, average="weighted")
    precision = load_metric("precision").compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": accuracy, "f1": f1, "precision": precision}

def find_max_length(dataset: list[str]) -> int:
    """Find the maximum length of the dataset.

    Args:
        dataset: The dataset.

    Returns:
        The maximum length of the dataset.
    """
    return len(max(dataset, key=lambda x: len(x.split())).split())

class HFSentimentAnalysisDataset(StrEnum):
    """HuggingFace Sentiment Analysis datasets."""
    financial_news = "zeroshot/twitter-financial-news-sentiment"
    imbd_reviews = "imdb"
    airline_reviews = "Shayanvsf/US_Airline_Sentiment"

class HFPretrainedModel(StrEnum):
    """HuggingFace Sentiment Analysis Model."""
    bert = "bert-base-uncased"
    roberta = "roberta-base"
    distilbert = "distilbert-base-cased"

class HFPretrainedTokenizer(StrEnum):
    """HuggingFace Sentiment Analysis datasets."""
    bert = "bert-base-uncased"
    roberta = "roberta-base"
    distilbert = "distilbert-base-cased"