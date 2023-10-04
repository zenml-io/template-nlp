# {% include 'template/license_header' %}

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from zenml.enums import StrEnum

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='micro')
    precision = precision_score(y_true=labels, y_pred=pred, average='micro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='micro')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}



def find_max_length(dataset):
    return len(max(dataset, key=lambda x: len(x.split())).split())


class HFSentimentAnalysisDataset(StrEnum):
    """HuggingFace Sentiment Analysis datasets."""
    financial_news = "zeroshot/twitter-financial-news-sentiment"
    imbd_reviews = "mushroomsolutions/imdb_sentiment_3000_Test"
    airline_reviews = "mattbit/tweet-sentiment-airlines"

class HFPretrainedModel(StrEnum):
    """HuggingFace Sentiment Analysis Model."""
    bert = "bert-base-uncased"
    gpt2 = "gpt2"

class HFPretrainedTokenizer(StrEnum):
    """HuggingFace Sentiment Analysis datasets."""
    bert = "bert-base-uncased"
    gpt2 = "gpt2"